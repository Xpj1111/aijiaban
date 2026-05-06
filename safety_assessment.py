# -*- coding: utf-8 -*-
"""
设备操作安全评估模块
防止危险操作，保护用户安全
"""

from typing import Dict, List, Optional, Tuple


class SafetyAssessment:
    """安全评估器"""
    
    # 设备安全参数定义
    DEVICE_SAFETY_LIMITS = {
        "热水器": {
            "safe_temp_range": (30, 60),  # 安全温度范围 °C
            "danger_temp_range": (60, 75),  # 危险温度范围（需要警告）
            "min_temp": 30,  # 最低允许温度
            "max_temp": 75,  # 最高允许温度
            "recommended_temp": 45,  # 推荐温度
            "risk_level": "high",  # 风险等级
            "warnings": [
                "水温超过60°C可能导致烫伤",
                "建议洗澡水温设置在40-45°C",
                "老人和儿童使用时应设置更低温度"
            ]
        },
        "空调": {
            "safe_temp_range": (20, 28),  # 舒适温度范围
            "danger_temp_range": None,
            "max_temp": 30,
            "min_temp": 16,
            "recommended_temp": 26,
            "risk_level": "low",
            "warnings": [
                "温度设置过低可能影响健康",
                "建议夏季设置在26°C左右"
            ]
        },
        "燃气灶": {
            "fire_levels": (1, 5),
            "risk_level": "high",
            "warnings": [
                "使用燃气灶时请确保通风",
                "离开厨房前请关闭燃气灶",
                "注意防火安全"
            ]
        },
        "灯光": {
            "brightness_range": (0, 100),
            "risk_level": "none",
            "warnings": []
        }
    }
    
    # 危险操作关键词
    DANGEROUS_ACTIONS = {
        "调高.*温度.*热水器",
        "提高.*水温",
        "加热.*水",
        "热水.*最高",
        "最大火力",
        "全开.*火"
    }
    
    @classmethod
    def assess_operation(cls, device: str, action: str, params: Dict) -> Dict:
        """
        评估操作的安全性
        
        Returns:
            {
                "is_safe": bool,           # 是否安全
                "risk_level": str,         # 风险等级: none/low/medium/high/critical
                "warnings": List[str],     # 警告信息
                "suggestions": List[str],  # 建议
                "should_confirm": bool,    # 是否需要二次确认
                "auto_adjusted": bool,     # 是否自动调整了参数
                "adjusted_params": Dict    # 调整后的参数
            }
        """
        result = {
            "is_safe": True,
            "risk_level": "none",
            "warnings": [],
            "suggestions": [],
            "should_confirm": False,
            "auto_adjusted": False,
            "adjusted_params": params.copy()
        }
        
        # 获取设备安全配置
        safety_config = cls.DEVICE_SAFETY_LIMITS.get(device)
        if not safety_config:
            return result
        
        result["risk_level"] = safety_config.get("risk_level", "none")
        result["warnings"] = safety_config.get("warnings", []).copy()
        
        # 检查热水器温度
        if device == "热水器" and action in ["set", "调节温度"]:
            temp_result = cls._check_water_heater_temperature(params, safety_config)
            result.update(temp_result)
        
        # 检查燃气灶火力
        elif device == "燃气灶" and action in ["set", "调节火力"]:
            fire_result = cls._check_stove_fire_level(params, safety_config)
            result.update(fire_result)
        
        # 检查空调温度
        elif device == "空调" and action in ["set", "调节温度"]:
            ac_result = cls._check_ac_temperature(params, safety_config)
            result.update(ac_result)
        
        return result
    
    @classmethod
    def _check_water_heater_temperature(cls, params: Dict, config: Dict) -> Dict:
        """检查热水器温度安全性"""
        result = {
            "is_safe": True,
            "should_confirm": False,
            "auto_adjusted": False,
            "adjusted_params": params.copy(),
            "warnings": config.get("warnings", []).copy(),
            "suggestions": []
        }
        
        target_temp = params.get("value") or params.get("temperature")
        if not target_temp:
            return result
        
        try:
            target_temp = float(target_temp)
        except (ValueError, TypeError):
            result["is_safe"] = False
            result["warnings"].append("无法解析温度值")
            return result
        
        # 超过最高温度 - 拒绝执行
        if target_temp > config["max_temp"]:
            result["is_safe"] = False
            result["warnings"].append(f"⚠️ 危险：温度 {target_temp}°C 超过最高限制 {config['max_temp']}°C")
            result["suggestions"].append(f"已自动调整为最高安全温度 {config['max_temp']}°C")
            result["auto_adjusted"] = True
            result["adjusted_params"]["value"] = config["max_temp"]
            result["adjusted_params"]["temperature"] = config["max_temp"]
            return result
        
        # 低于最低温度 - 拒绝执行并自动调整
        safe_min, safe_max = config["safe_temp_range"]
        if target_temp < safe_min:
            result["is_safe"] = False
            result["warnings"].append(f"⚠️ 危险：温度 {target_temp}°C 过低，可能导致热水器损坏或无法正常工作")
            result["suggestions"].append(f"热水器最低安全温度为 {safe_min}°C")
            result["suggestions"].append(f"已自动调整为最低安全温度 {safe_min}°C")
            result["auto_adjusted"] = True
            result["adjusted_params"]["value"] = safe_min
            result["adjusted_params"]["temperature"] = safe_min
            return result
        
        # 进入危险温度范围（过高）- 需要确认
        if config.get("danger_temp_range"):
            danger_min, danger_max = config["danger_temp_range"]
            if target_temp >= danger_min:
                result["should_confirm"] = True
                result["warnings"].append(
                    f"⚠️ 警告：温度 {target_temp}°C 较高，可能导致烫伤"
                )
                result["suggestions"].append(
                    f"建议使用安全温度 {config['recommended_temp']}°C"
                )
                return result
        
        # 在安全范围内，但偏低 - 给出建议
        if target_temp < safe_min + 5:  # 比最低安全温度高5度以内
            result["suggestions"].append(
                f"温度 {target_temp}°C 较低，洗澡可能会冷，建议设置为 {config['recommended_temp']}°C"
            )
        
        return result
    
    @classmethod
    def _check_stove_fire_level(cls, params: Dict, config: Dict) -> Dict:
        """检查燃气灶火力安全性"""
        result = {
            "is_safe": True,
            "should_confirm": False,
            "auto_adjusted": False,
            "adjusted_params": params.copy(),
            "warnings": config.get("warnings", []).copy(),
            "suggestions": []
        }
        
        fire_level = params.get("value") or params.get("level")
        if not fire_level:
            return result
        
        try:
            fire_level = int(fire_level)
        except (ValueError, TypeError):
            result["is_safe"] = False
            result["warnings"].append("无法解析火力等级")
            return result
        
        min_level, max_level = config["fire_levels"]
        
        # 超出范围
        if fire_level > max_level:
            result["is_safe"] = False
            result["warnings"].append(f"火力等级 {fire_level} 超过最大值 {max_level}")
            result["suggestions"].append(f"已自动调整为最大火力 {max_level}")
            result["auto_adjusted"] = True
            result["adjusted_params"]["value"] = max_level
            result["adjusted_params"]["level"] = max_level
        
        return result
    
    @classmethod
    def _check_ac_temperature(cls, params: Dict, config: Dict) -> Dict:
        """检查空调温度合理性"""
        result = {
            "is_safe": True,
            "should_confirm": False,
            "auto_adjusted": False,
            "adjusted_params": params.copy(),
            "warnings": config.get("warnings", []).copy(),
            "suggestions": []
        }
        
        target_temp = params.get("value") or params.get("temperature")
        if not target_temp:
            return result
        
        try:
            target_temp = float(target_temp)
        except (ValueError, TypeError):
            result["is_safe"] = False
            result["warnings"].append("无法解析温度值")
            return result
        
        # 超出设备限制
        if target_temp > config["max_temp"] or target_temp < config["min_temp"]:
            result["is_safe"] = False
            result["warnings"].append(
                f"温度 {target_temp}°C 超出设备范围 ({config['min_temp']}-{config['max_temp']}°C)"
            )
            # 自动调整到边界值
            adjusted = max(config["min_temp"], min(config["max_temp"], target_temp))
            result["suggestions"].append(f"已自动调整为 {adjusted}°C")
            result["auto_adjusted"] = True
            result["adjusted_params"]["value"] = adjusted
            result["adjusted_params"]["temperature"] = adjusted
        
        # 温度过低提醒
        elif target_temp < 20:
            result["suggestions"].append(
                f"温度 {target_temp}°C 较低，长时间使用可能影响健康"
            )
        
        return result
    
    @classmethod
    def generate_safety_report(cls, assessment: Dict) -> str:
        """生成安全报告文本"""
        if assessment["is_safe"] and not assessment["warnings"]:
            return ""
        
        report = "\n\n🛡️ 安全评估报告:\n"
        
        if not assessment["is_safe"]:
            report += "❌ 操作被阻止：存在安全隐患\n"
        
        if assessment["warnings"]:
            report += "⚠️ 警告:\n"
            for warning in assessment["warnings"]:
                report += f"   • {warning}\n"
        
        if assessment["suggestions"]:
            report += "💡 建议:\n"
            for suggestion in assessment["suggestions"]:
                report += f"   • {suggestion}\n"
        
        if assessment["should_confirm"]:
            report += "\n⚡ 此操作需要您的确认才能执行"
        
        if assessment["auto_adjusted"]:
            report += "\n✅ 系统已自动调整参数以确保安全"
        
        return report
