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


def get_texture_cache_path(input_root, input_path, output_root):
    """Mirror the source game's materials folder in the output tree."""
    from SourceIO.library.utils.path_utilities import get_mod_path
    from SourceIO.library.utils.tiny_path import TinyPath

    source_directory = os.path.dirname(os.path.abspath(input_path))
    game_root = os.path.abspath(str(get_mod_path(TinyPath(source_directory))))

    expected_game_dirs = ("materials", "maps", "models", "elements")
    if (
        not os.path.isdir(game_root)
        or not any(
            os.path.isdir(os.path.join(game_root, name))
            for name in expected_game_dirs
        )
    ):
        game_root = os.path.abspath(input_root)

    try:
        relative_game_root = os.path.relpath(game_root, os.path.abspath(input_root))
    except ValueError:
        relative_game_root = os.curdir

    if (
        relative_game_root == os.pardir
        or relative_game_root.startswith(os.pardir + os.sep)
    ):
        relative_game_root = os.curdir

    return os.path.abspath(os.path.join(output_root, relative_game_root, "materials"))


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
    """Discover game content and persist its mounts for SourceIO import operators."""
    from SourceIO.library.shared.content_manager import ContentManager
    from SourceIO.library.utils.tiny_path import TinyPath
    from SourceIO.blender_bindings.utils.resource_utils import serialize_mounted_content

    content_manager = ContentManager()
    content_manager.scan_for_content(TinyPath(input_root))

    serialize_mounted_content(content_manager)


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


def get_mtl_texture_path(line):
    """Extract a texture filename from an MTL map directive."""
    fields = line.strip().split()
    if len(fields) < 2:
        return None

    index = 1
    while index < len(fields) and fields[index].startswith("-"):
        option = fields[index]

        if option in ("-o", "-s"):
            index += 4
        elif option in (
            "-bm", "-boost", "-texres", "-clamp",
            "-blendu", "-blendv", "-imfchan", "-type",
        ):
            index += 2
        else:
            return None

    if index >= len(fields):
        return None

    return " ".join(fields[index:]).strip().strip('"')


def validate_obj_textures(output_path):
    """Require at least one diffuse MTL reference that resolves to a real PNG."""
    mtl_path = os.path.splitext(output_path)[0] + ".mtl"

    if not os.path.isfile(mtl_path):
        raise RuntimeError("Model export has no MTL material library.")

    with open(mtl_path, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            fields = line.strip().split(None, 1)
            if not fields or fields[0].casefold() != "map_kd":
                continue

            texture_reference = get_mtl_texture_path(line)
            if not texture_reference:
                continue

            if not os.path.isabs(texture_reference):
                texture_reference = os.path.join(os.path.dirname(mtl_path), texture_reference)

            texture_path = os.path.abspath(texture_reference)
            if not os.path.isfile(texture_path) or os.path.getsize(texture_path) == 0:
                continue

            with open(texture_path, "rb") as image_file:
                if image_file.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE:
                    print(
                        f"[model] Verified diffuse texture: {texture_path}",
                        flush=True,
                    )
                    return

    raise RuntimeError(
        "Model geometry exported, but its MTL has no valid map_Kd PNG "
        "reference. Texture import or material-to-image linking failed."
    )


def import_model(input_root, input_path, output_path, output_root):
    ensure_parent(output_path)

    model_directory = os.path.dirname(os.path.abspath(input_path))
    texture_cache_path = get_texture_cache_path(input_root, input_path, output_root)
    os.makedirs(texture_cache_path, exist_ok=True)

    bpy.context.scene.sourceio_props.TextureCachePath = texture_cache_path

    print(f"[model] SourceIO texture cache: {texture_cache_path}", flush=True)

    bpy.ops.sourceio.mdl(
        directory=model_directory,
        files=[{"name": os.path.basename(input_path)}],
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
        path_mode="RELATIVE",
    )

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Blender did not produce an OBJ file.")

    validate_obj_geometry(output_path)
    ensure_obj_material_library(output_path)
    validate_obj_textures(output_path)


def node_tree_has_image_texture(node_tree, visited=None):
    """Return True if a material node tree contains a usable image texture."""
    if node_tree is None:
        return False

    if visited is None:
        visited = set()

    pointer = node_tree.as_pointer()
    if pointer in visited:
        return False
    visited.add(pointer)

    for node in node_tree.nodes:
        if node.type == "TEX_IMAGE":
            image = node.image
            if image is not None and image.has_data:
                return True

        if node.type == "GROUP" and node_tree_has_image_texture(node.node_tree, visited):
            return True

    return False


def validate_bsp_scene():
    """Require usable BSP geometry, but never reject a map for missing textures."""
    mesh_objects = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH"
        and obj.data is not None
        and len(obj.data.vertices) > 0
        and len(obj.data.polygons) > 0
    ]

    if not mesh_objects:
        raise RuntimeError("BSP import produced no usable mesh geometry.")

    materials = {
        slot.material
        for obj in mesh_objects
        for slot in obj.material_slots
        if slot.material is not None
    }

    textured_materials = [
        material for material in materials
        if material.use_nodes
        and node_tree_has_image_texture(material.node_tree)
    ]

    print(
        f"[map] Geometry OK: {len(mesh_objects)} mesh objects; "
        f"{len(materials)} assigned materials; "
        f"{len(textured_materials)} textured materials.",
        flush=True,
    )

    if not textured_materials:
        print(
            "[map] WARNING: no usable image-textured materials were found. "
            "Saving the map anyway so its geometry is preserved.",
            flush=True,
        )


def import_bsp(input_root, input_path, output_path, output_root):
    if os.path.splitext(input_path)[1].casefold() != ".bsp":
        raise ValueError("Only an explicit .bsp source may create a .blend output.")

    ensure_parent(output_path)

    texture_cache_path = get_texture_cache_path(input_root, input_path, output_root)
    os.makedirs(texture_cache_path, exist_ok=True)
    bpy.context.scene.sourceio_props.TextureCachePath = texture_cache_path
    
    bpy.ops.sourceio.bsp(
        filepath=input_path,
        discover_resources=True,
        import_textures=True,
        use_bvlg=False,
    )
    validate_bsp_scene()
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


def ensure_sourceio_registered():
    if hasattr(bpy.types.Scene, "sourceio_props"):
        return

    import importlib.util

    print(
        "[startup] Locating the installed SourceIO add-on...",
        flush=True,
    )

    addon_dirs = list(bpy.utils.script_paths(subdir="addons"))
    user_addons_dir = bpy.utils.user_resource(
        "SCRIPTS",
        path="addons",
    )

    if user_addons_dir:
        addon_dirs.insert(0, user_addons_dir)

    addon_root = None
    searched_dirs = []

    for addons_dir in dict.fromkeys(addon_dirs):
        searched_dirs.append(addons_dir)

        if not os.path.isdir(addons_dir):
            continue

        try:
            entries = list(os.scandir(addons_dir))
        except OSError:
            continue

        for entry in entries:
            if not entry.is_dir():
                continue
            if "sourceio" not in entry.name.casefold():
                continue

            candidate = entry.path

            if (
                os.path.isfile(os.path.join(candidate, "__init__.py"))
                and os.path.isfile(
                    os.path.join(
                        candidate,
                        "blender_bindings",
                        "bindings.py",
                    )
                )
            ):
                addon_root = candidate
                break

        if addon_root is not None:
            break

    if addon_root is None:
        raise RuntimeError(
            "Could not locate SourceIO's add-on folder. "
            "Searched: " + "; ".join(searched_dirs)
        )

    init_path = os.path.join(addon_root, "__init__.py")

    print(f"[startup] Loading SourceIO from: {addon_root}", flush=True)

    spec = importlib.util.spec_from_file_location(
        "SourceIO",
        init_path,
        submodule_search_locations=[addon_root],
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not create a Python module spec for: {init_path}"
        )

    sourceio_module = importlib.util.module_from_spec(spec)
    sys.modules["SourceIO"] = sourceio_module

    spec.loader.exec_module(sourceio_module)

    from SourceIO.blender_bindings.bindings import (
        register as register_sourceio,
    )

    register_sourceio()

    try:
        scene_props = bpy.context.scene.sourceio_props
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            "SourceIO register() returned, but the active scene "
            "does not expose sourceio_props."
        ) from exc

    if scene_props is None:
        raise RuntimeError("SourceIO sourceio_props is None on the active scene.")

    print("[startup] SourceIO registered successfully.", flush=True)


def main():
    if "--" not in sys.argv:
        raise RuntimeError("Expected a manifest path after '--'.")

    manifest_path = sys.argv[sys.argv.index("--") + 1]
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    input_root = os.path.abspath(manifest["input_root"])
    output_root = os.path.abspath(manifest["output_root"])
    jobs = manifest["jobs"]
    sourceio_registered = False

    if jobs:
        ensure_sourceio_registered()
        sourceio_registered = True

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
                import_model(input_root, input_path, output_path, output_root)
            elif asset_type == "map":
                import_bsp(input_root, input_path, output_path, output_root)
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

    if sourceio_registered:
        try:
            from SourceIO.blender_bindings.bindings import unregister_custom_icon

            unregister_custom_icon()
        except Exception as exc:
            print(
                f"[shutdown] Warning: could not release SourceIO icon previews: {exc}",
                flush=True,
            )

    result_path = os.path.splitext(manifest_path)[0] + ".results.json"
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()