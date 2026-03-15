"""
fix_font.py - 修复 Windows 上 matplotlib 中文显示问题

运行一次即可，之后所有图表中文都能正常显示。
运行方式：python fix_font.py
"""

import matplotlib
import matplotlib.pyplot as plt
import os

def fix_chinese_font():
    # Windows 上可用的中文字体，按优先级排列
    chinese_fonts = [
        "Microsoft YaHei",   # 微软雅黑（Win7+自带）
        "SimHei",            # 黑体
        "SimSun",            # 宋体
        "KaiTi",             # 楷体
        "FangSong",          # 仿宋
    ]

    # 检查哪个字体可用
    from matplotlib import font_manager
    available = [f.name for f in font_manager.fontManager.ttflist]

    chosen = None
    for font in chinese_fonts:
        if font in available:
            chosen = font
            print(f"✓ 找到可用中文字体：{font}")
            break

    if not chosen:
        print("⚠ 未找到预设中文字体，尝试搜索系统字体...")
        # 尝试直接从 Windows 字体目录加载
        win_font_dir = r"C:\Windows\Fonts"
        candidates = {
            "msyh.ttc":   "Microsoft YaHei",
            "msyhbd.ttc": "Microsoft YaHei",
            "simhei.ttf": "SimHei",
            "simsun.ttc": "SimSun",
        }
        for fname, fname_label in candidates.items():
            fpath = os.path.join(win_font_dir, fname)
            if os.path.exists(fpath):
                font_manager.fontManager.addfont(fpath)
                chosen = fname_label
                print(f"✓ 手动加载字体：{fpath}")
                break

    if not chosen:
        print("❌ 未找到任何中文字体，请确认系统安装了微软雅黑或黑体")
        return False

    # 写入 matplotlib 配置文件（永久生效）
    config_dir  = matplotlib.get_configdir()
    config_path = os.path.join(config_dir, "matplotlibrc")

    config_content = f"""# matplotlib 中文字体配置（由 fix_font.py 自动生成）
font.family         : sans-serif
font.sans-serif     : {chosen}, DejaVu Sans, Arial
axes.unicode_minus  : False
"""

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)

    print(f"✓ 配置已写入：{config_path}")

    # 验证
    plt.rcParams["font.family"]       = ["sans-serif"]
    plt.rcParams["font.sans-serif"]   = [chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 画一张测试图
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_title("中文字体测试：差分隐私深度学习")
    ax.set_xlabel("隐私预算 ε")
    ax.set_ylabel("测试准确率")
    ax.plot([1, 2, 3], [0.85, 0.90, 0.95], "o-")
    test_path = "font_test.png"
    plt.savefig(test_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"✓ 测试图已保存：{test_path}（请打开确认中文是否正常）")
    return True


if __name__ == "__main__":
    print("=" * 45)
    print("matplotlib 中文字体修复工具")
    print("=" * 45)
    success = fix_chinese_font()
    if success:
        print("\n✅ 修复完成！重新运行 step4_final_report.py 和 exp_visualize.py 即可。")
    else:
        print("\n请手动在代码顶部加入：")
        print("  import matplotlib")
        print("  matplotlib.rcParams['font.sans-serif'] = ['SimHei']")
        print("  matplotlib.rcParams['axes.unicode_minus'] = False")
