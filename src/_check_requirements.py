import pkg_resources

packages = {
    "pandas":       "2.2.2",
    "numpy":        "1.26.4",
    "scikit-learn": "1.5.0",
    "statsmodels":  "0.14.2",
    "xgboost":      "2.0.3",
    "matplotlib":   "3.9.0",
    "seaborn":      "0.13.2",
    "folium":       "0.16.0",
    "pyarrow":      "16.1.0",
    "astral":       "3.2",
    "contextily":   "1.7.0",
    "pyproj":       "3.7.1",
    "python-docx":  None,   # 未鎖版本
}

print(f"{'套件':<20} {'requirements.txt':<18} {'實際安裝':<18} 狀態")
print("-" * 72)

all_ok = True
for pkg, req_ver in packages.items():
    try:
        installed = pkg_resources.get_distribution(pkg).version
    except pkg_resources.DistributionNotFound:
        installed = "NOT INSTALLED"

    req_str = req_ver if req_ver else "(未鎖版本)"

    if installed == "NOT INSTALLED":
        status = "❌ 未安裝"
        all_ok = False
    elif req_ver is None:
        status = "WARN: version not pinned"
    elif installed == req_ver:
        status = "OK"
    else:
        status = f"WARN: expect {req_ver}"
        all_ok = False

    print(f"{pkg:<20} {req_str:<18} {installed:<18} {status}")

print()
if all_ok:
    print("結論：requirements.txt 涵蓋所有套件，版本一致 ✓")
else:
    print("結論：有套件需要修正（見上方 ❌）")
