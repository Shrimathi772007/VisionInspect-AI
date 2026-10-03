p = "app/ai/evaluation/model_family_final.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''                "threshold_over_normal_loio_max": None, "normal_loio_distribution": normals,''', '''                "normal_loio_distribution": normals,''')
s = s.replace('''    position.pop("threshold_over_normal_loio_max")\n''', '')
s = s.replace('''        "4_selection_reproduced_from_the_saved_normal_and_synthetic_study_only": True,''', '''        "4_selection_reproduced_from_the_saved_normal_and_synthetic_study_only": mfs.select_configuration(
            {c: r["summary"] for c, r in study["candidates"].items()}, mfs.registry())["winner"] == f"{cid}|{lock['selected_policy']}",''')
open(p, "w", encoding="utf-8").write(s)
print("patched", "position.pop" not in s)
