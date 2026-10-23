import bpy
import json
import os
import sys
import traceback


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


def import_vtf(input_root, input_path, output_path):
    ensure_parent(output_path)

    relative_path = get_relative_input_path(
        input_root,
        input_path,
    )

    image_names_before = { image.name for image in bpy.data.images }

    bpy.ops.sourceio.vtf(
        directory=input_root,
        files=[{"name": relative_path}],
    )

    new_images = [
        image
        for image in bpy.data.images
        if image.name not in image_names_before
    ]

    image = new_images[-1] if new_images else None

    if image is None:
        stem = os.path.splitext(os.path.basename(input_path))[0].casefold()

        for candidate in reversed(list(bpy.data.images)):
            full_path = candidate.get("full_path", "")

            if os.path.basename(full_path).casefold() == (os.path.basename(input_path).casefold()):
                image = candidate
                break

            if candidate.name.casefold() == stem:
                image = candidate
                break

    if image is None:
        raise RuntimeError("SourceIO did not produce a discoverable image.")

    image.filepath_raw = output_path
    image.file_format = "PNG"
    image.save()


def import_vmt(input_root, input_path, output_path):
    ensure_parent(output_path)

    relative_path = get_relative_input_path(
        input_root,
        input_path,
    )

    bpy.ops.sourceio.vmt(
        directory=input_root,
        files=[{"name": relative_path}],
        discover_resources=True,
    )

    pack_used_images()

    bpy.ops.wm.save_as_mainfile(
        filepath=output_path,
        check_existing=False,
    )


def import_model(input_root, input_path, output_path):
    ensure_parent(output_path)

    relative_path = get_relative_input_path(
        input_root,
        input_path,
    )

    bpy.ops.sourceio.mdl(
        directory=input_root,
        files=[{"name": relative_path}],
        discover_resources=True,
        import_textures=True,
        import_animations=False,
    )

    bpy.ops.object.select_all(action="SELECT")
    selected = list(bpy.context.selected_objects)

    if not selected:
        raise RuntimeError("SourceIO imported no model objects.")

    bpy.ops.wm.obj_export(
        filepath=output_path,
        export_selected_objects=True,
        export_materials=True,
    )


def import_bsp(input_root, input_path, output_path):
    ensure_parent(output_path)

    from SourceIO.library.shared.content_manager import ContentManager
    from SourceIO.library.utils.tiny_path import TinyPath

    content_manager = ContentManager()
    content_manager.scan_for_content(TinyPath(input_root))

    bpy.ops.sourceio.bsp(
        filepath=input_path,
        discover_resources=True,
    )

    pack_used_images()

    bpy.ops.wm.save_as_mainfile(
        filepath=output_path,
        check_existing=False,
    )


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
            clear_scene()

            if asset_type == "texture":
                import_vtf(
                    input_root,
                    input_path,
                    output_path,
                )
            elif asset_type == "material":
                import_vmt(
                    input_root,
                    input_path,
                    output_path,
                )
            elif asset_type == "model":
                import_model(
                    input_root,
                    input_path,
                    output_path,
                )
            elif asset_type == "map":
                import_bsp(
                    input_root,
                    input_path,
                    output_path,
                )
            else:
                raise ValueError(f"Unknown conversion type: {asset_type}")

            results.append({
                "InputPath": input_path,
                "OutputPaths": [output_path],
                "Status": 0,
                "Message": None,
            })

        except Exception as exc:
            traceback.print_exc()

            results.append({
                "InputPath": input_path,
                "OutputPaths": [],
                "Status": 3,
                "Message": str(exc),
            })

    result_path = os.path.splitext(manifest_path)[0] + ".results.json"

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    if any(result["Status"] == 3 for result in results):
        print("Some assets failed to convert.", flush=True)


if __name__ == "__main__":
    main()