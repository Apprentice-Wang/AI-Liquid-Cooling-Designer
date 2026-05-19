import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def mm_to_m(value_mm):
    return value_mm / 1000.0


def l_min_to_m3_s(value_l_min):
    return value_l_min / 1000.0 / 60.0


def load_case(case_path):
    with open(case_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_single_case(case):
    """
    一个简化的液冷冷板热设计计算模型。

    注意：
    这不是严格 CFD 仿真，而是用于早期设计筛选的工程估算模型。
    """

    # -----------------------------
    # 1. 读取输入参数
    # -----------------------------
    chip_length = mm_to_m(case["chip"]["length_mm"])
    chip_width = mm_to_m(case["chip"]["width_mm"])
    chip_power = case["chip"]["power_W"]

    inlet_temp = case["coolant"]["inlet_temperature_C"]
    flow_rate = l_min_to_m3_s(case["coolant"]["volume_flow_rate_L_min"])

    channel_number = case["channels"]["number"]
    channel_width = mm_to_m(case["channels"]["width_mm"])
    channel_height = mm_to_m(case["channels"]["height_mm"])
    channel_length = mm_to_m(case["channels"]["length_mm"])

    plate_k = case["cold_plate"]["thermal_conductivity_W_mK"]
    plate_thickness = mm_to_m(case["cold_plate"]["thickness_mm"])

    # -----------------------------
    # 2. 冷却液物性参数：水，约 25 ℃
    # -----------------------------
    rho = 997.0          # kg/m3
    mu = 0.00089         # Pa·s
    k_water = 0.6        # W/m/K
    cp = 4182.0          # J/kg/K
    pr = cp * mu / k_water

    # -----------------------------
    # 3. 几何与流动参数
    # -----------------------------
    single_channel_area = channel_width * channel_height
    total_flow_area = channel_number * single_channel_area

    velocity = flow_rate / total_flow_area

    hydraulic_diameter = 2 * channel_width * channel_height / (
        channel_width + channel_height
    )

    reynolds = rho * velocity * hydraulic_diameter / mu

    # -----------------------------
    # 4. 努塞尔数 Nu 与换热系数 h
    # -----------------------------
    if reynolds < 2300:
        flow_regime = "Laminar"
        nu = 4.36
    else:
        flow_regime = "Turbulent"
        nu = 0.023 * reynolds ** 0.8 * pr ** 0.4

    h = nu * k_water / hydraulic_diameter

    # -----------------------------
    # 5. 热阻估算
    # -----------------------------
    chip_area = chip_length * chip_width

    conduction_resistance = plate_thickness / (plate_k * chip_area)

    # 简化认为换热面积为所有流道底面和侧壁面积
    heat_transfer_area = channel_number * (
        channel_width + 2 * channel_height
    ) * channel_length

    convection_resistance = 1.0 / (h * heat_transfer_area)

    total_thermal_resistance = conduction_resistance + convection_resistance

    # 冷却液平均温升
    mass_flow_rate = rho * flow_rate
    coolant_temperature_rise = chip_power / (mass_flow_rate * cp)

    # 芯片最高温度简化估算
    max_temperature = inlet_temp + coolant_temperature_rise + chip_power * total_thermal_resistance

    # -----------------------------
    # 6. 压降估算
    # -----------------------------
    if reynolds < 2300:
        friction_factor = 64.0 / reynolds
    else:
        friction_factor = 0.3164 / reynolds ** 0.25

    pressure_drop_pa = friction_factor * (
        channel_length / hydraulic_diameter
    ) * 0.5 * rho * velocity ** 2

    pressure_drop_kpa = pressure_drop_pa / 1000.0

    pumping_power = pressure_drop_pa * flow_rate

    result = {
        "flow_regime": flow_regime,
        "velocity_m_s": velocity,
        "hydraulic_diameter_mm": hydraulic_diameter * 1000,
        "Re": reynolds,
        "Pr": pr,
        "Nu": nu,
        "h_W_m2K": h,
        "conduction_resistance_K_W": conduction_resistance,
        "convection_resistance_K_W": convection_resistance,
        "total_thermal_resistance_K_W": total_thermal_resistance,
        "coolant_temperature_rise_C": coolant_temperature_rise,
        "max_temperature_C": max_temperature,
        "pressure_drop_kPa": pressure_drop_kpa,
        "pumping_power_W": pumping_power,
    }

    return result


def print_result(result):
    print("\n========== Liquid Cooling Design Result ==========\n")

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\n==================================================\n")


def generate_flow_rate_study(case):
    """
    改变流量，观察最高温度和压降的变化。
    """

    flow_rates = np.linspace(0.2, 3.0, 30)

    records = []

    for flow in flow_rates:
        new_case = yaml.safe_load(yaml.dump(case))
        new_case["coolant"]["volume_flow_rate_L_min"] = float(flow)

        result = calculate_single_case(new_case)

        records.append({
            "flow_rate_L_min": flow,
            "max_temperature_C": result["max_temperature_C"],
            "pressure_drop_kPa": result["pressure_drop_kPa"],
            "thermal_resistance_K_W": result["total_thermal_resistance_K_W"],
            "pumping_power_W": result["pumping_power_W"],
            "Re": result["Re"],
        })

    return pd.DataFrame(records)


def plot_results(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 图1：流量 vs 最高温度
    plt.figure(figsize=(6, 4))
    plt.plot(df["flow_rate_L_min"], df["max_temperature_C"], marker="o")
    plt.xlabel("Volume flow rate / L min$^{-1}$")
    plt.ylabel("Maximum temperature / °C")
    plt.title("Effect of flow rate on maximum temperature")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "flow_rate_vs_temperature.png", dpi=300)
    plt.close()

    # 图2：流量 vs 压降
    plt.figure(figsize=(6, 4))
    plt.plot(df["flow_rate_L_min"], df["pressure_drop_kPa"], marker="o")
    plt.xlabel("Volume flow rate / L min$^{-1}$")
    plt.ylabel("Pressure drop / kPa")
    plt.title("Effect of flow rate on pressure drop")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "flow_rate_vs_pressure_drop.png", dpi=300)
    plt.close()

    # 图3：温度-压降权衡图
    plt.figure(figsize=(6, 4))
    plt.scatter(df["pressure_drop_kPa"], df["max_temperature_C"])
    plt.xlabel("Pressure drop / kPa")
    plt.ylabel("Maximum temperature / °C")
    plt.title("Thermal-hydraulic trade-off")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "thermal_hydraulic_tradeoff.png", dpi=300)
    plt.close()

    # 图4：示意温度云图，不是真实 CFD
    x = np.linspace(0, 40, 120)
    y = np.linspace(0, 40, 120)
    X, Y = np.meshgrid(x, y)

    base_temp = 35
    hotspot = 35 * np.exp(-((X - 22) ** 2 + (Y - 21) ** 2) / (2 * 8 ** 2))
    cooling_gradient = -8 * (X / 40)
    T = base_temp + hotspot + cooling_gradient

    plt.figure(figsize=(6, 5))
    plt.imshow(T, origin="lower", extent=[0, 40, 0, 40], aspect="equal")
    plt.colorbar(label="Temperature / °C")
    plt.xlabel("Chip length / mm")
    plt.ylabel("Chip width / mm")
    plt.title("Conceptual temperature field")
    plt.tight_layout()
    plt.savefig(output_dir / "conceptual_temperature_field.png", dpi=300)
    plt.close()


def main():
    case_path = Path("examples/single_chip_case.yaml")
    output_dir = Path("figures")
    data_dir = Path("data")

    output_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    if not case_path.exists():
        raise FileNotFoundError(
            "Cannot find examples/single_chip_case.yaml. "
            "Please create the YAML case file first."
        )

    case = load_case(case_path)

    result = calculate_single_case(case)
    print_result(result)

    df = generate_flow_rate_study(case)

    df.to_csv(data_dir / "flow_rate_study.csv", index=False)

    plot_results(df, output_dir)

    print("Generated files:")
    print("1. data/flow_rate_study.csv")
    print("2. figures/flow_rate_vs_temperature.png")
    print("3. figures/flow_rate_vs_pressure_drop.png")
    print("4. figures/thermal_hydraulic_tradeoff.png")
    print("5. figures/conceptual_temperature_field.png")


if __name__ == "__main__":
    main()
