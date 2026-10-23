import bpy
import json
import os
import sys
import traceback

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
BLEND_SIGNATURE = b"BLENDER"
DEFAULT_MATERIAL = "ValveConvert_Default"


def clear_scene():
    if bpy.context.scene.objects:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

    for collection_name in (
        "meshes",
        "curves",
        "armatures",
        "materials",
        "images",
        "cameras",
        "lights",
    ):
        collection = getattr(bpy.data, collection_name, None)
        if collection is None:
            continue

        for datablock in list(collection):
            if datablock.users == 0:
                collection.remove(datablock)


def ensure_parent(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def get_relative_input_path(input_root, input_path):
    root = os.path.abspath(input_root)
    path = os.path.abspath(input_path)
    relative = os.path.relpath(path, root)

    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        raise ValueError(f"Input file is outside the input directory: {input_path}")

    return relative.replace("\\", "/")


def pack_used_images():
    for image in bpy.data.images:
        if image.users <= 0 or not image.has_data:
            continue
        try:
            if image.packed_file is None:
                image.pack()
        except Exception as exc:
            print(f"Warning: unable to pack image {image.name}: {exc}", flush=True)


def scan_source_content(input_root):
    from SourceIO.library.shared.content_manager import ContentManager
    from SourceIO.library.utils.tiny_path import TinyPath

    content_manager = ContentManager()
    content_manager.scan_for_content(TinyPath(input_root))


def import_vtf(input_root, input_path, output_path):
    ensure_parent(output_path)
    relative_path = get_relative_input_path(input_root, input_path)
    image_names_before = {image.name for image in bpy.data.images}

    bpy.ops.sourceio.vtf(
        directory=input_root,
        files=[{"name": relative_path}],
    )

    new_images = [image for image in bpy.data.images if image.name not in image_names_before]
    image = new_images[-1] if new_images else None

    if image is None:
        wanted_name = os.path.basename(input_path).casefold()
        wanted_stem = os.path.splitext(wanted_name)[0]
        for candidate in reversed(list(bpy.data.images)):
            full_path = candidate.get("full_path", "")
            if os.path.basename(full_path).casefold() == wanted_name:
                image = candidate
                break
            if candidate.name.casefold() == wanted_stem:
                image = candidate
                break

    if image is None:
        raise RuntimeError("SourceIO did not produce a discoverable image.")

    image.filepath_raw = output_path
    image.file_format = "PNG"
    image.save()

    with open(output_path, "rb") as file:
        if file.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
            raise RuntimeError("The texture exporter did not produce a valid PNG file.")


def validate_obj_geometry(output_path):
    has_vertices = False
    has_faces = False

    with open(output_path, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                has_vertices = True
            elif parts[0] == "f" and len(parts) >= 4:
                has_faces = True
            if has_vertices and has_faces:
                return

    raise RuntimeError("Model import/export produced no usable mesh geometry (OBJ must contain vertices and faces).")


def ensure_obj_material_library(output_path):
    """Ensure the OBJ references a corresponding non-empty MTL with every used material."""
    mtl_path = os.path.splitext(output_path)[0] + ".mtl"
    mtl_name = os.path.basename(mtl_path)

    with open(output_path, "r", encoding="utf-8", errors="replace") as file:
        lines = file.read().splitlines()

    used_materials = []
    for line in lines:
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] == "usemtl" and parts[1] not in used_materials:
            used_materials.append(parts[1])

    if not used_materials:
        first_face = next(
            (index for index, line in enumerate(lines)
             if line.strip().split()[:1] == ["f"]),
            None,
        )
        if first_face is None:
            raise RuntimeError("OBJ contains no faces; refusing to save an empty model export.")
        used_materials = [DEFAULT_MATERIAL]
        lines.insert(first_face, "usemtl " + DEFAULT_MATERIAL)

    present_materials = set()
    if os.path.isfile(mtl_path) and os.path.getsize(mtl_path) > 0:
        with open(mtl_path, "r", encoding="utf-8", errors="replace") as file:
            for line in file:
                parts = line.strip().split(None, 1)
                if len(parts) == 2 and parts[0] == "newmtl":
                    present_materials.add(parts[1])

    missing_materials = [name for name in used_materials if name not in present_materials]
    if missing_materials:
        with open(mtl_path, "a", encoding="utf-8", newline="\n") as file:
            if os.path.getsize(mtl_path) > 0:
                file.write("\n")
            for name in missing_materials:
                file.write(
                    f"newmtl {name}\n"
                    "Ns 20.000000\n"
                    "Ka 0.800000 0.800000 0.800000\n"
                    "Kd 0.800000 0.800000 0.800000\n"
                    "Ks 0.000000 0.000000 0.000000\n"
                    "d 1.000000\n"
                    "illum 2\n\n"
                )

    lines = [line for line in lines if not line.strip().startswith("mtllib ")]
    lines.insert(0, "mtllib " + mtl_name)
    with open(output_path, "w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines) + "\n")

    if not os.path.isfile(mtl_path) or os.path.getsize(mtl_path) == 0:
        raise RuntimeError("OBJ export did not produce a non-empty MTL file.")


def import_model(input_root, input_path, output_path):
    ensure_parent(output_path)
    relative_path = get_relative_input_path(input_root, input_path)

    scan_source_content(input_root)
    bpy.ops.sourceio.mdl(
        directory=input_root,
        files=[{"name": relative_path}],
        discover_resources=True,
        import_textures=True,
        import_animations=False,
    )

    mesh_objects = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH"
        and obj.data is not None
        and len(obj.data.vertices) > 0
        and len(obj.data.polygons) > 0
    ]
    if not mesh_objects:
        raise RuntimeError("SourceIO imported no mesh geometry for this MDL. Check its matching VVD/VTX files.")

    fallback = bpy.data.materials.get(DEFAULT_MATERIAL)
    if fallback is None:
        fallback = bpy.data.materials.new(DEFAULT_MATERIAL)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
        if len(obj.data.materials) == 0:
            obj.data.materials.append(fallback)

    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.wm.obj_export(
        filepath=output_path,
        export_selected_objects=True,
        export_materials=True,
    )

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Blender did not produce an OBJ file.")

    validate_obj_geometry(output_path)
    ensure_obj_material_library(output_path)


def import_bsp(input_root, input_path, output_path):
    if os.path.splitext(input_path)[1].casefold() != ".bsp":
        raise ValueError("Only an explicit .bsp source may create a .blend output.")

    ensure_parent(output_path)
    scan_source_content(input_root)
    bpy.ops.sourceio.bsp(
        filepath=input_path,
        discover_resources=True,
    )
    pack_used_images()
    bpy.ops.wm.save_as_mainfile(filepath=output_path, check_existing=False)

    with open(output_path, "rb") as file:
        if file.read(len(BLEND_SIGNATURE)) != BLEND_SIGNATURE:
            raise RuntimeError("BSP conversion did not produce a valid Blender project file.")


def validate_output(asset_type, output_path):
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Converter returned without creating a non-empty output file.")

    extension = os.path.splitext(output_path)[1].casefold()
    if extension == ".blend" and asset_type != "map":
        raise RuntimeError("Safety check rejected a .blend output not originating from a BSP job.")
    if asset_type == "map" and extension != ".blend":
        raise RuntimeError("A BSP job must produce a .blend file.")
    if asset_type == "texture" and extension != ".png":
        raise RuntimeError("A texture job must produce a .png file.")
    if asset_type == "model" and extension != ".obj":
        raise RuntimeError("A model job must produce an .obj file.")


def remove_failed_outputs(asset_type, output_path):
    paths = [output_path]
    if asset_type == "model":
        paths.append(os.path.splitext(output_path)[0] + ".mtl")
    for path in paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError as exc:
            print(f"Warning: could not remove failed output {path}: {exc}", flush=True)


def main():
    if "--" not in sys.argv:
        raise RuntimeError("Expected a manifest path after '--'.")

    manifest_path = sys.argv[sys.argv.index("--") + 1]
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    input_root = os.path.abspath(manifest["input_root"])
    jobs = manifest["jobs"]
    results = []

    for job in jobs:
        asset_type = job.get("type", job.get("Type"))
        input_path = job.get("input_path", job.get("InputPath"))
        output_path = job.get("output_path", job.get("OutputPath"))
        print(f"[{asset_type}] {input_path}", flush=True)

        try:
            extension = os.path.splitext(input_path)[1].casefold()
            output_extension = os.path.splitext(output_path)[1].casefold()
            if output_extension == ".blend" and not (asset_type == "map" and extension == ".bsp"):
                raise ValueError("Only an explicit .bsp input may create a .blend output.")
            if asset_type == "map" and extension != ".bsp":
                raise ValueError("A map conversion job must point to an explicit .bsp file.")

            clear_scene()
            if asset_type == "texture":
                import_vtf(input_root, input_path, output_path)
            elif asset_type == "model":
                import_model(input_root, input_path, output_path)
            elif asset_type == "map":
                import_bsp(input_root, input_path, output_path)
            else:
                raise ValueError(f"Unsupported conversion job type: {asset_type}. VMT-to-BLEND is disabled.")

            validate_output(asset_type, output_path)
            output_paths = [output_path]
            if asset_type == "model":
                mtl_path = os.path.splitext(output_path)[0] + ".mtl"
                if not os.path.isfile(mtl_path) or os.path.getsize(mtl_path) == 0:
                    raise RuntimeError("Model conversion did not produce its required MTL file.")
                output_paths.append(mtl_path)

            results.append({
                "InputPath": input_path,
                "OutputPaths": output_paths,
                "Status": 0,
                "Message": None,
            })

        except Exception as exc:
            traceback.print_exc()
            remove_failed_outputs(asset_type, output_path)
            results.append({
                "InputPath": input_path,
                "OutputPaths": [],
                "Status": 3,
                "Message": str(exc),
            })

    result_path = os.path.splitext(manifest_path)[0] + ".results.json"
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)