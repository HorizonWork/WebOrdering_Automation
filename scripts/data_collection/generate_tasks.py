# -*- coding: utf-8 -*-
"""
Sinh task list cho Lazada & Shopee:
- scripts/data_collection/tasks_lazada.yaml  (150 task)
- scripts/data_collection/tasks_shopee.yaml (150 task)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import yaml  # pip install pyyaml nếu chưa có


PRODUCT_TYPES = [
    "điện thoại", "laptop", "tai nghe bluetooth", "loa bluetooth", "máy tính bảng",
    "màn hình máy tính", "chuột không dây", "bàn phím cơ", "ổ cứng SSD", "RAM máy tính",
    "tủ lạnh", "máy giặt", "máy lạnh", "nồi chiên không dầu", "nồi cơm điện",
    "bếp điện từ", "ấm đun siêu tốc", "máy lọc không khí", "robot hút bụi", "máy hút bụi",
    "áo thun nam", "áo sơ mi nam", "quần jean nữ", "váy công sở", "giày thể thao nam",
    "giày sneaker nữ", "balo laptop", "vali kéo", "túi xách nữ", "ví da nam",
    "kem chống nắng", "sữa rửa mặt", "son dưỡng môi", "nước hoa nữ", "nước hoa nam",
    "sữa tắm em bé", "bỉm tã cho bé", "sữa công thức cho bé", "ghế ăn dặm", "xe đẩy em bé",
    "bàn học sinh", "ghế văn phòng", "đèn bàn học", "kệ sách", "ga giường",
    "chăn lông", "gối ngủ", "thảm trải sàn", "rèm cửa", "tranh treo tường",
    "đồng hồ treo tường", "đồng hồ thông minh", "vòng đeo tay theo dõi sức khỏe",
    "cân điện tử", "máy đo huyết áp",
    "áo khoác chống nắng nữ", "áo hoodie nam", "quần jogger nam", "áo len nữ", "chân váy dài",
    "quạt điện", "quạt điều hòa", "bộ dao nhà bếp", "bộ nồi inox", "bộ chén đĩa sứ",
    "bút bi", "bút máy", "sổ tay", "máy in", "mực in",
    "camera wifi", "chuông cửa thông minh", "ổ cắm điện thông minh", "bộ phát wifi",
    "dây sạc điện thoại",
    "pin dự phòng", "miếng dán cường lực", "ốp lưng điện thoại", "bàn phím bluetooth",
    "loa soundbar",
    "áo bra thể thao", "quần legging tập gym", "thảm tập yoga", "bình nước thể thao",
    "găng tay tập gym",
    "bộ đồ ngủ nữ", "đèn ngủ để bàn", "nến thơm", "tinh dầu lavender",
    "máy khuếch tán tinh dầu",
    "áo polo nam", "quần short nam", "quần short nữ", "đầm maxi đi biển", "kính mát nam",
    "kính mát nữ", "mũ bảo hiểm nửa đầu", "mũ bảo hiểm fullface", "găng tay đi xe máy",
    "áo mưa bộ",
    "ổ cắm đa năng", "dây nối ổ cứng", "webcam cho máy tính", "micro thu âm",
    "tai nghe chụp tai gaming",
    "bàn phím gaming", "ghế gaming", "card màn hình", "mainboard máy tính", "CPU Intel",
    "điện thoại iPhone", "điện thoại Samsung", "điện thoại Xiaomi", "điện thoại Oppo",
    "điện thoại Realme",
]

COLORS = ["đỏ", "đen", "trắng", "xanh dương", "xanh lá", "hồng", "xám", "nâu"]
BUDGETS = ["200k", "300k", "500k", "800k", "1 triệu", "2 triệu", "3 triệu", "5 triệu", "10 triệu"]
REGIONS = ["Hà Nội", "TP.HCM", "Đà Nẵng", "Cần Thơ"]


def gen_lazada_tasks(n: int = 150) -> List[Dict[str, str]]:
    tasks: List[Dict[str, str]] = []

    templates = [
        "Tìm {prod} giá khoảng {budget} trên Lazada",
        "Mua {prod} chính hãng, ưu tiên Mall, giá dưới {budget} trên Lazada",
        "Tìm {prod} màu {color} cho nữ, giá khoảng {budget} trên Lazada",
        "Tìm {prod} phù hợp làm quà tặng, đóng gói đẹp trên Lazada",
        "Mua {prod} giao nhanh trong ngày tại {region} trên Lazada",
        "Tìm {prod} có nhiều đánh giá 5 sao, miễn phí vận chuyển trên Lazada",
        "Mua {prod} tiết kiệm điện, độ bền cao cho gia đình 4 người trên Lazada",
        "Tìm {prod} cỡ nhỏ, phù hợp phòng trọ sinh viên trên Lazada",
        "Mua {prod} cho người lớn tuổi, dễ sử dụng, chữ to trên Lazada",
        "Tìm {prod} bản mới nhất năm nay, bảo hành ít nhất 12 tháng trên Lazada",
    ]

    i = 0
    while len(tasks) < n:
        prod = PRODUCT_TYPES[i % len(PRODUCT_TYPES)]
        tmpl = templates[i % len(templates)]
        budget = BUDGETS[i % len(BUDGETS)]
        color = COLORS[i % len(COLORS)]
        region = REGIONS[i % len(REGIONS)]
        query = tmpl.format(prod=prod, budget=budget, color=color, region=region)
        tasks.append({"query": query, "url": "https://www.lazada.vn"})
        i += 1

    return tasks


def gen_shopee_tasks(n: int = 150) -> List[Dict[str, str]]:
    tasks: List[Dict[str, str]] = []

    templates = [
        "Tìm {prod} giá rẻ, ưu tiên nhiều mã giảm giá trên Shopee",
        "Mua {prod} chính hãng, có hóa đơn VAT trên Shopee",
        "Tìm {prod} màu {color}, phù hợp đi làm văn phòng trên Shopee",
        "Mua {prod} giá khoảng {budget}, ưu tiên Freeship Xtra trên Shopee",
        "Tìm {prod} giao nhanh 2h tại {region} trên Shopee",
        "Mua {prod} có lượt bán trên 1k, đánh giá từ 4.8 trở lên trên Shopee",
        "Tìm {prod} cho sinh viên, giá mềm nhưng dùng ổn trên Shopee",
        "Mua {prod} combo nhiều món để tiết kiệm chi phí trên Shopee",
        "Tìm {prod} bản nội địa Trung, review tốt trên Shopee",
        "Mua {prod} của shop yêu thích+ để tích điểm trên Shopee",
    ]

    i = 0
    while len(tasks) < n:
        prod = PRODUCT_TYPES[i % len(PRODUCT_TYPES)]
        tmpl = templates[i % len(templates)]
        budget = BUDGETS[i % len(BUDGETS)]
        color = COLORS[i % len(COLORS)]
        region = REGIONS[i % len(REGIONS)]
        query = tmpl.format(prod=prod, budget=budget, color=color, region=region)
        tasks.append({"query": query, "url": "https://shopee.vn"})
        i += 1

    return tasks


def main() -> None:
    out_dir = Path("scripts/data_collection")
    out_dir.mkdir(parents=True, exist_ok=True)

    lazada_tasks = gen_lazada_tasks(150)
    shopee_tasks = gen_shopee_tasks(150)

    lazada_path = out_dir / "tasks_lazada.yaml"
    shopee_path = out_dir / "tasks_shopee.yaml"

    lazada_path.write_text(
        yaml.dump(lazada_tasks, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    shopee_path.write_text(
        yaml.dump(shopee_tasks, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print(f"Wrote {len(lazada_tasks)} Lazada tasks to {lazada_path}")
    print(f"Wrote {len(shopee_tasks)} Shopee tasks to {shopee_path}")


if __name__ == "__main__":
    main()
