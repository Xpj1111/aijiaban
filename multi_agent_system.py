# -*- coding: utf-8 -*-
"""
多Agent协作系统
主Agent + 安全Agent + 执行Agent 协同工作
"""

import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path


class AgentMessage:
    """Agent间通信消息"""
    
    def __init__(self, sender: str, receiver: str, message_type: str, content: Dict):
        self.sender = sender  # 发送者
        self.receiver = receiver  # 接收者
        self.message_type = message_type  # 消息类型
        self.content = content  # 消息内容
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.message_id = f"msg_{int(datetime.now().timestamp() * 1000)}"
    
    def to_dict(self) -> Dict:
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp
        }


class BaseAgent:
    """Agent基类"""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.message_history: List[AgentMessage] = []
    
    def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """接收消息并处理，返回响应消息"""
        self.message_history.append(message)
        response = self.process_message(message)
        if response:
            self.message_history.append(response)
        return response
    
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理消息 - 子类实现"""
        raise NotImplementedError
    
    def send_message(self, receiver: str, message_type: str, content: Dict) -> AgentMessage:
        """创建发送消息"""
        return AgentMessage(
            sender=self.name,
            receiver=receiver,
            message_type=message_type,
            content=content
        )


class MainAgent(BaseAgent):
    """
    主Agent - 协调者
    职责：
    1. 理解用户意图
    2. 分解任务
    3. 协调子Agent工作
    4. 整合结果并回复用户
    """
    
    def __init__(self):
        super().__init__("MainAgent", "Coordinator")
        self.agents_registry = {}
        # 初始化模型管理器
        try:
            from inference.model_manager import ModelManager
            
            # 创建简化配置
            class SimpleConfig:
                model_type = "gguf"
                gguf_path = "./models/qwen2.5-3b-instruct-q4_k_m.gguf"
                download_url = ""
                
                class Inference:
                    n_ctx = 32768
                    n_threads = 4
                    n_gpu_layers = 0
                    temperature = 0.7
                    top_p = 0.9
                    max_tokens = 512
                
                inference = Inference()
            
            config = SimpleConfig()
            self.model_manager = ModelManager(config)
            self.model_manager.initialize()
            print("✅ MainAgent: 大模型初始化成功")
        except Exception as e:
            print(f"⚠️ MainAgent: 大模型初始化失败，将使用规则匹配模式: {e}")
            self.model_manager = None
    
    def register_agent(self, agent: BaseAgent):
        """注册子Agent"""
        self.agents_registry[agent.name] = agent
    
    def process_user_request(self, user_input: str) -> Dict:
        """
        处理用户请求的完整流程
        
        流程：
        1. 分析用户意图
        2. 如果是设备控制，委托给SafetyAgent评估
        3. 如果安全，委托给ExecutionAgent执行
        4. 整合结果返回
        """
        print(f"\n{'='*70}")
        print(f"🤖 MainAgent: 收到用户请求")
        print(f"{'='*70}")
        print(f"用户: {user_input}\n")
        
        # Step 1: 意图识别
        intent = self._analyze_intent(user_input)
        print(f"💭 MainAgent思考: 用户意图是 '{intent['type']}'")
        print(f"   详细描述: {intent['description']}\n")
        
        # Step 2: 根据意图类型分发任务
        if intent["type"] == "device_control":
            return self._handle_device_control(user_input, intent)
        elif intent["type"] == "reminder":
            return self._handle_reminder(user_input, intent)
        else:
            return self._handle_chat(user_input, intent)
    
    def _analyze_intent(self, user_input: str) -> Dict:
        """分析用户意图（使用大模型）"""
        if self.model_manager:
            return self._analyze_intent_with_llm(user_input)
        else:
            return self._analyze_intent_with_rules(user_input)
    
    def _analyze_intent_with_llm(self, user_input: str) -> Dict:
        """使用大模型分析用户意图"""
        system_prompt = """你是一个智能家居意图识别专家。请分析用户的输入，识别其意图类型。

支持的意图类型：
1. device_control - 设备控制（打开/关闭/调节家电设备）
2. reminder - 提醒设置（设置闹钟、提醒事项等）
3. chat - 普通对话（闲聊、问答等）

请以JSON格式返回结果，包含以下字段：
- type: 意图类型（device_control/reminder/chat）
- device: 如果是设备控制，提取设备名称（热水器/空调/灯/电视/窗帘/燃气灶/冰箱等），否则为null
- action: 如果是设备控制，提取操作（on/off/set），否则为null
- value: 如果有具体数值（如温度），提取该值，否则为null
- description: 对用户意图的简要描述
- confidence: 置信度（0-1之间）

示例：
用户输入："把热水器调到45度"
返回：{"type": "device_control", "device": "热水器", "action": "set", "value": 45, "description": "用户想要调节热水器温度到45度", "confidence": 0.95}

用户输入："打开空调"
返回：{"type": "device_control", "device": "空调", "action": "on", "value": null, "description": "用户想要打开空调", "confidence": 0.98}

用户输入："明天早上8点提醒我开会"
返回：{"type": "reminder", "device": null, "action": null, "value": null, "description": "用户想要设置明天早上8点的会议提醒", "confidence": 0.92}

用户输入："你好"
返回：{"type": "chat", "device": null, "action": null, "value": null, "description": "普通问候对话", "confidence": 0.99}"""
        
        try:
            result_text = self.model_manager.chat(
                prompt=user_input,
                system_prompt=system_prompt,
                temperature=0.3,  # 降低温度以提高稳定性
                max_tokens=256
            )
            
            # 尝试解析JSON结果
            import json
            import re
            
            # 提取JSON部分（可能包含在markdown代码块中）
            json_match = re.search(r'\{[^}]+\}', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                intent = json.loads(json_str)
                
                # 验证必要字段
                if "type" in intent and intent["type"] in ["device_control", "reminder", "chat"]:
                    intent["raw_input"] = user_input
                    print(f"🤖 LLM意图识别结果: {intent['type']} (置信度: {intent.get('confidence', 'N/A')})")
                    return intent
            
            # 如果解析失败，降级到规则匹配
            print("⚠️ LLM结果解析失败，降级到规则匹配")
            return self._analyze_intent_with_rules(user_input)
            
        except Exception as e:
            print(f"⚠️ LLM意图识别出错: {e}，降级到规则匹配")
            return self._analyze_intent_with_rules(user_input)
    
    def _analyze_intent_with_rules(self, user_input: str) -> Dict:
        """基于规则的意图分析（降级方案）"""
        user_lower = user_input.lower()
        
        # 设备控制关键词
        device_keywords = ["热水器", "空调", "灯", "电视", "窗帘", "燃气灶", "冰箱"]
        action_keywords = ["打开", "关闭", "调节", "设置", "调到", "开启"]
        
        has_device = any(kw in user_input for kw in device_keywords)
        has_action = any(kw in user_input for kw in action_keywords)
        
        if has_device and has_action:
            # 提取设备和操作
            device = None
            for kw in device_keywords:
                if kw in user_input:
                    device = kw
                    break
            
            return {
                "type": "device_control",
                "device": device,
                "description": f"用户想要控制{device}",
                "raw_input": user_input
            }
        
        # 提醒相关
        if any(kw in user_lower for kw in ["提醒", "闹钟", "记住"]):
            return {
                "type": "reminder",
                "description": "用户想要设置提醒",
                "raw_input": user_input
            }
        
        # 普通对话
        return {
            "type": "chat",
            "description": "普通对话",
            "raw_input": user_input
        }
    
    def _handle_device_control(self, user_input: str, intent: Dict) -> Dict:
        """处理设备控制请求"""
        print("🔄 MainAgent: 检测到设备控制请求，启动安全评估流程...\n")
        
        # Step 1: 解析操作参数
        operation = self._parse_device_operation(user_input, intent)
        print(f"📋 MainAgent: 解析的操作参数:")
        print(f"   设备: {operation['device']}")
        print(f"   操作: {operation['action']}")
        print(f"   参数: {operation.get('value', 'N/A')}\n")
        
        # Step 2: 发送给SafetyAgent进行安全评估
        safety_agent = self.agents_registry.get("SafetyAgent")
        if not safety_agent:
            return {"error": "SafetyAgent未注册"}
        
        safety_request = self.send_message(
            receiver="SafetyAgent",
            message_type="safety_assessment",
            content={
                "operation": operation,
                "user_input": user_input
            }
        )
        
        print("📨 MainAgent → SafetyAgent: 请求安全评估")
        safety_response = safety_agent.receive_message(safety_request)
        
        if not safety_response:
            return {"error": "安全评估失败"}
        
        assessment = safety_response.content
        print(f"\n📥 MainAgent ← SafetyAgent: 收到评估结果")
        print(f"   安全状态: {'✅ 安全' if assessment['is_safe'] else '❌ 不安全'}")
        print(f"   风险等级: {assessment['risk_level']}\n")
        
        # Step 3: 根据评估结果决策
        if not assessment["is_safe"]:
            # 不安全，拒绝执行
            print("❌ MainAgent: 操作被SafetyAgent阻止\n")
            return {
                "intent": "device_control",
                "thinking": f"用户请求操作{operation['device']}，但SafetyAgent评估认为存在安全隐患：{'; '.join(assessment['warnings'])}",
                "actions": [],
                "response": self._generate_safety_denial_response(assessment, operation),
                "safety_assessment": assessment
            }
        
        # 如果需要确认，也应该拒绝（等待用户明确确认）
        if assessment.get("should_confirm"):
            print("⚡ MainAgent: 操作需要用户确认，暂时拒绝执行\n")
            return {
                "intent": "device_control",
                "thinking": f"用户请求操作{operation['device']}，SafetyAgent评估认为存在一定风险，需要用户明确确认。",
                "actions": [],
                "response": self._generate_confirmation_request(assessment, operation),
                "safety_assessment": assessment
            }
        
        # Step 4: 安全，发送给ExecutionAgent执行
        execution_agent = self.agents_registry.get("ExecutionAgent")
        if not execution_agent:
            return {"error": "ExecutionAgent未注册"}
        
        exec_request = self.send_message(
            receiver="ExecutionAgent",
            message_type="execute_operation",
            content={
                "operation": assessment.get("adjusted_params", operation),
                "assessment": assessment
            }
        )
        
        print("📨 MainAgent → ExecutionAgent: 请求执行操作")
        exec_response = execution_agent.receive_message(exec_request)
        
        if not exec_response:
            return {"error": "执行失败"}
        
        execution_result = exec_response.content
        print(f"\n📥 MainAgent ← ExecutionAgent: 收到执行结果")
        print(f"   执行状态: {'✅ 成功' if execution_result['success'] else '❌ 失败'}\n")
        
        # Step 5: 整合结果
        return {
            "intent": "device_control",
            "thinking": f"用户请求控制{operation['device']}。经SafetyAgent评估安全后，由ExecutionAgent执行。",
            "actions": [{
                "action": "control_device",
                "params": operation,
                "description": execution_result.get("description", "")
            }],
            "response": execution_result.get("response", "操作已完成"),
            "safety_assessment": assessment,
            "execution_result": execution_result
        }
    
    def _handle_reminder(self, user_input: str, intent: Dict) -> Dict:
        """处理提醒请求（简化）"""
        return {
            "intent": "reminder",
            "thinking": "用户想要设置提醒",
            "actions": [],
            "response": "提醒功能需要进一步实现"
        }
    
    def _handle_chat(self, user_input: str, intent: Dict) -> Dict:
        """处理普通对话"""
        return {
            "intent": "chat",
            "thinking": "普通对话，无需特殊处理",
            "actions": [],
            "response": f"我理解您的意思：{user_input}。有什么我可以帮您的吗？"
        }
    
    def _parse_device_operation(self, user_input: str, intent: Dict) -> Dict:
        """解析设备操作参数（简化版）"""
        import re
        
        device = intent.get("device", "")
        
        # 提取温度值
        temp_match = re.search(r'(\d+)\s*(?:度|°C)', user_input)
        value = int(temp_match.group(1)) if temp_match else None
        
        # 判断操作类型
        if any(kw in user_input for kw in ["打开", "开启", "开"]):
            action = "on"
        elif any(kw in user_input for kw in ["关闭", "关掉", "关"]):
            action = "off"
        else:
            action = "set"
        
        return {
            "device": device,
            "action": action,
            "value": value
        }
    
    def _generate_confirmation_request(self, assessment: Dict, operation: Dict) -> str:
        """生成需要确认的回复"""
        response = f"⚠️ 此操作可能存在风险，需要您的确认。\n\n"
        
        if assessment["warnings"]:
            response += "风险提示:\n"
            for warning in assessment["warnings"]:
                response += f"  • {warning}\n"
        
        if assessment.get("suggestions"):
            response += f"\n💡 建议:\n"
            for suggestion in assessment["suggestions"]:
                response += f"  • {suggestion}\n"
        
        adjusted = assessment.get("adjusted_params", {})
        if adjusted and adjusted.get("value"):
            response += f"\n✅ 我已将参数调整为更安全的值: {adjusted['value']}\n"
            response += f"   如果您确认使用这个安全值，请说'确认'或'是的'。"
        else:
            response += f"\n请确认是否继续执行？"
        
        return response
    
    def _generate_safety_denial_response(self, assessment: Dict, operation: Dict) -> str:
        """生成安全拒绝的回复"""
        response = f"⚠️ 为了您的安全，我无法执行此操作。\n\n"
        
        if assessment["warnings"]:
            response += "原因:\n"
            for warning in assessment["warnings"]:
                response += f"  • {warning}\n"
        
        if assessment.get("auto_adjusted"):
            adjusted = assessment["adjusted_params"]
            response += f"\n💡 建议：我已将参数调整为更安全的值 {adjusted.get('value', '')}\n"
            response += f"   如果您确认要使用这个安全值，请告诉我。"
        else:
            response += f"\n💡 建议：请选择一个更安全的参数值。"
        
        return response
    
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理来自其他Agent的消息"""
        # MainAgent通常不接收其他Agent的主动消息
        return None


class SafetyAgent(BaseAgent):
    """
    安全Agent - 安全专家
    职责：
    1. 评估操作的安全性（使用LLM智能评估）
    2. 识别潜在风险
    3. 提供安全建议
    4. 必要时自动调整参数
    """
    
    def __init__(self):
        super().__init__("SafetyAgent", "Security Expert")
        # 初始化模型管理器用于智能安全评估
        try:
            from inference.model_manager import ModelManager
            
            class SimpleConfig:
                model_type = "gguf"
                gguf_path = "./models/qwen2.5-3b-instruct-q4_k_m.gguf"
                download_url = ""
                
                class Inference:
                    n_ctx = 32768
                    n_threads = 4
                    n_gpu_layers = 0
                    temperature = 0.3  # 低温度保证稳定性
                    top_p = 0.9
                    max_tokens = 512
                
                inference = Inference()
            
            config = SimpleConfig()
            self.model_manager = ModelManager(config)
            self.model_manager.initialize()
            print("✅ SafetyAgent: 大模型初始化成功")
        except Exception as e:
            print(f"⚠️ SafetyAgent: 大模型初始化失败，将使用规则评估模式: {e}")
            self.model_manager = None
            # 降级到规则评估
            from safety_assessment import SafetyAssessment
            self.assessor = SafetyAssessment
    
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理安全评估请求"""
        if message.message_type != "safety_assessment":
            return None
        
        print(f"🛡️  SafetyAgent: 开始安全评估")
        
        operation = message.content["operation"]
        user_input = message.content.get("user_input", "")
        
        print(f"   评估对象: {operation['device']}")
        print(f"   操作类型: {operation['action']}")
        print(f"   参数值: {operation.get('value', 'N/A')}")
        
        # 使用LLM进行智能安全评估
        if self.model_manager:
            assessment = self._assess_with_llm(operation, user_input)
        else:
            # 降级到规则评估
            assessment = self.assessor.assess_operation(
                device=operation["device"],
                action=operation["action"],
                params=operation
            )
        
        print(f"\n📊 SafetyAgent评估结果:")
        print(f"   安全状态: {'✅ 安全' if assessment['is_safe'] else '❌ 不安全'}")
        print(f"   风险等级: {assessment['risk_level']}")
        
        if assessment["warnings"]:
            print(f"   警告数量: {len(assessment['warnings'])}")
        
        if assessment["auto_adjusted"]:
            print(f"   ⚙️  已自动调整参数")
        
        # 生成详细报告
        if self.model_manager:
            report = self._generate_safety_report_llm(assessment)
        else:
            report = self.assessor.generate_safety_report(assessment)
        if report:
            print(report)
        
        # 返回评估结果
        return self.send_message(
            receiver="MainAgent",
            message_type="assessment_result",
            content=assessment
        )
    
    def _assess_with_llm(self, operation: Dict, user_input: str) -> Dict:
        """使用大模型进行智能安全评估"""
        system_prompt = """你是一个智能家居安全评估专家。请评估用户设备操作的安全性。

## 评估原则

### 热水器安全标准
- **安全温度范围**: 30-60°C
- **推荐温度**: 40-45°C（洗澡）
- **危险温度**: >60°C（烫伤风险）
- **最高限制**: 75°C
- **特殊人群**: 老人、儿童应使用38-42°C

### 空调安全标准
- **舒适范围**: 20-28°C
- **推荐温度**: 夏季26°C，冬季22°C
- **最低限制**: 16°C
- **最高限制**: 30°C

### 燃气灶安全标准
- **火力等级**: 1-5档
- **安全提醒**: 使用时保持通风，离开前关闭

### 通用安全原则
1. **人身安全优先**: 任何可能伤害用户的操作都应拒绝或调整
2. **设备保护**: 避免损坏设备的极端参数
3. **能源节约**: 建议使用节能参数
4. **特殊关怀**: 考虑老人、儿童等特殊群体

## 输出格式

你必须只返回一个JSON对象，不要包含任何其他文字。格式如下：
{
  "is_safe": true或false,
  "risk_level": "none"或"low"或"medium"或"high"或"critical",
  "warnings": ["警告信息1", "警告信息2"],
  "suggestions": ["建议1", "建议2"],
  "should_confirm": true或false,
  "auto_adjusted": true或false,
  "adjusted_params": {"device": "设备名", "action": "操作", "value": 调整后的值},
  "reasoning": "评估理由说明"
}

## 示例

用户操作：{"device": "热水器", "action": "set", "value": 70}
返回：
{"is_safe": false, "risk_level": "high", "warnings": ["70度水温过高，有严重烫伤风险", "超过安全上限60度"], "suggestions": ["建议设置为45度，这是舒适的洗澡温度", "已自动调整为45度"], "should_confirm": false, "auto_adjusted": true, "adjusted_params": {"device": "热水器", "action": "set", "value": 45}, "reasoning": "用户设置的70度远超安全范围，存在烫伤风险，已自动调整到安全的45度"}

用户操作：{"device": "热水器", "action": "set", "value": 45}
返回：
{"is_safe": true, "risk_level": "none", "warnings": [], "suggestions": [], "should_confirm": false, "auto_adjusted": false, "adjusted_params": {"device": "热水器", "action": "set", "value": 45}, "reasoning": "45度在安全范围内，是推荐的洗澡温度"}"""
        
        try:
            # 构建评估请求
            prompt = f"""请评估以下设备操作的安全性：

用户原始输入: {user_input}
操作参数: {json.dumps(operation, ensure_ascii=False)}

请给出安全评估结果（只返回JSON，不要其他文字）。"""
            
            result_text = self.model_manager.chat(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=512
            )
            
            # 尝试多种方式解析JSON
            import re
            
            # 方法1: 直接尝试解析整个响应
            try:
                assessment = json.loads(result_text.strip())
                print(f"🤖 LLM安全评估成功 (直接解析)")
            except:
                # 方法2: 提取第一个JSON对象
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    assessment = json.loads(json_str)
                    print(f"🤖 LLM安全评估成功 (正则提取)")
                else:
                    raise ValueError("无法从响应中提取JSON")
            
            # 验证必要字段
            required_fields = ["is_safe", "risk_level", "warnings", "suggestions"]
            if all(field in assessment for field in required_fields):
                # 补充默认值
                assessment.setdefault("should_confirm", False)
                assessment.setdefault("auto_adjusted", False)
                assessment.setdefault("adjusted_params", operation.copy())
                assessment.setdefault("reasoning", "")
                
                print(f"   风险等级: {assessment['risk_level']} (安全: {assessment['is_safe']})")
                return assessment
            else:
                raise ValueError(f"JSON缺少必要字段: {required_fields}")
            
        except Exception as e:
            print(f"⚠️ LLM安全评估出错: {e}，降级到规则评估")
            from safety_assessment import SafetyAssessment
            return SafetyAssessment.assess_operation(
                device=operation["device"],
                action=operation["action"],
                params=operation
            )
    
    def _generate_safety_report_llm(self, assessment: Dict) -> str:
        """生成LLM评估的安全报告"""
        if assessment["is_safe"] and not assessment["warnings"]:
            return ""
        
        report = "\n\n🛡️ 智能安全评估报告:\n"
        
        if not assessment["is_safe"]:
            report += "❌ 操作被阻止：存在安全隐患\n"
        
        if assessment.get("reasoning"):
            report += f"\n💭 评估分析:\n   {assessment['reasoning']}\n"
        
        if assessment["warnings"]:
            report += "\n⚠️ 警告:\n"
            for warning in assessment["warnings"]:
                report += f"   • {warning}\n"
        
        if assessment["suggestions"]:
            report += "\n💡 建议:\n"
            for suggestion in assessment["suggestions"]:
                report += f"   • {suggestion}\n"
        
        if assessment["should_confirm"]:
            report += "\n⚡ 此操作需要您的确认才能执行"
        
        if assessment["auto_adjusted"]:
            adjusted_value = assessment["adjusted_params"].get("value", "")
            report += f"\n✅ 系统已自动调整参数以确保安全 (调整为: {adjusted_value})"
        
        return report


class ExecutionAgent(BaseAgent):
    """
    执行Agent - 操作执行者
    职责：
    1. 执行具体的设备控制操作
    2. 验证执行结果
    3. 反馈执行状态
    """
    
    def __init__(self):
        super().__init__("ExecutionAgent", "Operator")
        self.execution_log = []
    
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理执行请求"""
        if message.message_type != "execute_operation":
            return None
        
        print(f"⚙️  ExecutionAgent: 开始执行操作")
        
        operation = message.content["operation"]
        assessment = message.content.get("assessment", {})
        
        print(f"   执行设备: {operation['device']}")
        print(f"   执行操作: {operation['action']}")
        print(f"   参数值: {operation.get('value', 'N/A')}")
        
        # 模拟执行（实际应该调用真实设备接口）
        success, description = self._execute_device_control(operation)
        
        print(f"\n✅ ExecutionAgent执行完成:")
        print(f"   状态: {'成功' if success else '失败'}")
        print(f"   描述: {description}")
        
        # 记录执行日志
        self.execution_log.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operation": operation,
            "success": success,
            "description": description
        })
        
        # 生成友好回复
        response = self._generate_response(operation, success, assessment)
        
        return self.send_message(
            receiver="MainAgent",
            message_type="execution_result",
            content={
                "success": success,
                "description": description,
                "response": response,
                "operation": operation
            }
        )
    
    def _execute_device_control(self, operation: Dict) -> tuple:
        """执行设备控制（模拟）"""
        device = operation["device"]
        action = operation["action"]
        value = operation.get("value")
        
        # 这里应该调用真实的设备控制接口
        # 目前只是模拟
        if action == "on":
            return True, f"{device}已开启"
        elif action == "off":
            return True, f"{device}已关闭"
        elif action == "set" and value is not None:
            return True, f"{device}已设置为{value}"
        else:
            return False, f"无法执行操作: {action}"
    
    def _generate_response(self, operation: Dict, success: bool, assessment: Dict) -> str:
        """生成执行结果的友好回复"""
        device = operation["device"]
        action = operation["action"]
        value = operation.get("value")
        
        if not success:
            return f"❌ 操作失败，请稍后重试"
        
        # 根据操作类型生成不同回复
        if action == "on":
            return f"✅ 已开启{device}"
        elif action == "off":
            return f"✅ 已关闭{device}"
        elif action == "set" and value is not None:
            # 如果有安全评估的调整，说明一下
            if assessment.get("auto_adjusted"):
                original_value = assessment.get("original_value", value)
                return (f"✅ 已将{device}设置为{value}。\n"
                       f"   （原请求{original_value}，为确保安全已调整）")
            else:
                return f"✅ 已将{device}设置为{value}"
        
        return f"✅ 操作已完成"


class MultiAgentSystem:
    """
    多Agent系统管理器
    负责初始化、协调和管理所有Agent
    """
    
    def __init__(self):
        print("=" * 70)
        print("🏗️  初始化多Agent协作系统")
        print("=" * 70)
        
        # 创建Agent实例
        self.main_agent = MainAgent()
        self.safety_agent = SafetyAgent()
        self.execution_agent = ExecutionAgent()
        
        # 注册子Agent到MainAgent
        self.main_agent.register_agent(self.safety_agent)
        self.main_agent.register_agent(self.execution_agent)
        
        print("✅ MainAgent 已创建")
        print("✅ SafetyAgent 已创建")
        print("✅ ExecutionAgent 已创建")
        print("✅ Agent注册完成\n")
    
    def process_request(self, user_input: str) -> Dict:
        """处理用户请求的入口"""
        return self.main_agent.process_user_request(user_input)
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            "agents": {
                "main": {
                    "name": self.main_agent.name,
                    "messages_processed": len(self.main_agent.message_history)
                },
                "safety": {
                    "name": self.safety_agent.name,
                    "assessments_count": len([m for m in self.safety_agent.message_history 
                                             if m.message_type == "safety_assessment"])
                },
                "execution": {
                    "name": self.execution_agent.name,
                    "executions_count": len(self.execution_agent.execution_log)
                }
            }
        }
