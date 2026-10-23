import bpy
import json
import os
import sys
import traceback

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for datablock in list(bpy.data.collections):
        if datablock.name != "Collection" and datablock.users == 0:
            bpy.data.collections.remove(datablock)

def ensure_parent(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

def import_vtf(input_path, output_path):
    ensure_parent(output_path)

    bpy.ops.sourceio.vtf(
        directory=os.path.dirname(input_path),
        files=[{"name": os.path.basename(input_path)}],
    )

    image = bpy.data.images.get(os.path.splitext(os.path.basename(input_path))[0])

    if image is None:
        candidates = [
            item for item in bpy.data.images
            if item.get("full_path", "").lower() == os.path.basename(input_path).lower()
        ]
        if candidates:
            image = candidates[0]
    if image is None:
        raise RuntimeError("SourceIO imported no discoverable image.")

    image.filepath_raw = output_path
    image.file_format = "PNG"
    image.save()


def import_model(input_path, output_path):
    ensure_parent(output_path)

    bpy.ops.sourceio.mdl(
        directory=os.path.dirname(input_path),
        files=[{"name": os.path.basename(input_path)}],
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


def import_bsp(input_path, output_path, steam_app_id):
    ensure_parent(output_path)

    bpy.ops.sourceio.bsp(
        filepath=input_path,
        discover_resources=True,
        steam_app_id=str(steam_app_id),
    )

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

    jobs = manifest["jobs"]
    steam_app_id = manifest.get("steam_app_id", 400)
    results = []

    for job in jobs:
        asset_type = job.get("type", job.get("Type"))
        input_path = job.get("input_path", job.get("InputPath"))
        output_path = job.get("output_path", job.get("OutputPath"))

        print(f"[{asset_type}] {input_path}", flush=True)

        try:
            clear_scene()

            if asset_type == "texture":
                import_vtf(input_path, output_path)
            elif asset_type == "model":
                import_model(input_path, output_path)
            elif asset_type == "map":
                import_bsp(input_path, output_path, steam_app_id)
            else:
                raise ValueError(f"Unknown asset type: {asset_type}")

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