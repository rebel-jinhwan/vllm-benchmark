#!/usr/bin/env python3
"""
并发多会话回放测试工具
模拟真实场景下多用户同时使用的情况，用于：
1. 压力测试：测试API在并发负载下的表现
2. 限流测试：测试API的速率限制和并发控制
3. 真实场景模拟：评估多用户场景下的性能和稳定性
4. 成本估算：预估多用户场景下的Token消耗和费用
"""

import asyncio
import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI
    import anthropic
    import httpx
except ImportError as e:
    logger.error(f"缺少依赖库: {e}")
    logger.error("请安装: pip install openai anthropic httpx")
    sys.exit(1)

# 可视化库（可选）
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib未安装，将无法生成matplotlib图表")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly未安装，将无法生成plotly图表")

# 导入random用于随机选择
import random


@dataclass
class ConcurrentSessionMetrics:
    """并发会话的性能指标"""
    session_id: str
    session_index: int  # 会话编号（用于标识）
    title: str
    total_turns: int
    tested_turns: int  # 实际测试的轮数

    # 时间指标
    start_time: str
    end_time: str
    total_duration_ms: float

    # Token指标
    total_input_tokens: int
    total_output_tokens: int

    # 成功率
    successful_turns: int
    failed_turns: int
    success_rate: float

    # TTFT和TPS指标（新增）
    avg_ttft_ms: Optional[float] = None
    median_ttft_ms: Optional[float] = None
    avg_tps: Optional[float] = None
    median_tps: Optional[float] = None

    # 有效样本数（用于加权平均）
    valid_tps_samples: int = 0
    valid_ttft_samples: int = 0

    # 每轮对话的详细数据（新增）
    turn_details: List[Dict[str, Any]] = field(default_factory=list)

    # 错误信息
    errors: List[str] = field(default_factory=list)


@dataclass
class ConcurrentTestReport:
    """并发测试报告"""
    provider_name: str
    model_name: str
    api_url: str
    test_time: str

    # 并发配置
    total_sessions: int
    max_concurrent_sessions: int
    max_turns_per_session: Optional[int]

    # 总体统计
    total_test_duration_ms: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    overall_success_rate: float

    # Token统计
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int

    # 性能指标
    requests_per_second: float  # QPS
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float

    # 每个会话的结果
    sessions: List[ConcurrentSessionMetrics]

    # 错误汇总
    total_errors: int

    # TTFT和TPS指标（新增）
    avg_ttft_ms: Optional[float] = None
    median_ttft_ms: Optional[float] = None
    p95_ttft_ms: Optional[float] = None
    avg_tps: Optional[float] = None
    median_tps: Optional[float] = None
    error_types: Dict[str, int] = field(default_factory=dict)


class ConcurrentTester:
    """并发测试器"""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        provider_name: str = "Unknown",
        api_format: str = "anthropic",
        repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None
    ):
        """初始化测试器"""
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.provider_name = provider_name
        self.api_format = api_format
        self.use_raw_httpx = False
        # 重复惩罚参数
        self.repetition_penalty = repetition_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        # 初始化客户端
        if api_format == "anthropic":
            if '/v1/messages' in api_url:
                base_url = api_url.rsplit('/v1/messages', 1)[0]
            elif api_url.endswith('/messages'):
                base_url = api_url.rsplit('/messages', 1)[0]
            else:
                base_url = api_url

            if 'anthropic.com' not in base_url:
                self.use_raw_httpx = True
                self.httpx_client = httpx.AsyncClient(timeout=300.0)
                logger.info(f"  ⚙️  使用原生 httpx客户端（第三方API）")
            else:
                self.client = anthropic.AsyncAnthropic(
                    api_key=api_key,
                    base_url=base_url
                )
        else:
            # OpenAI SDK会自动添加/chat/completions，所以需要去掉
            base_url = api_url
            if base_url.endswith('/chat/completions'):
                base_url = base_url.rsplit('/chat/completions', 1)[0]

            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )

    def load_sessions_data(self, json_file: str) -> Dict[str, Any]:
        """加载多会话数据"""
        path = Path(json_file)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {json_file}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"📂 加载数据: {json_file}")
        logger.info(f"  会话数: {data.get('total_sessions', len(data.get('sessions', [])))}")

        return data

    def select_sessions(
        self,
        sessions: List[Dict[str, Any]],
        num_sessions: Optional[int] = None,
        selection_mode: str = 'first',
        random_seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        选择要测试的sessions
        Args:
            sessions: 所有会话列表
            num_sessions: 要选择的会话数量（None表示全部）
            selection_mode: 选择模式 ('first': 前N个, 'random': 随机N个)
            random_seed: 随机种子（用于可重复的随机选择）
        Returns:
            选择后的会话列表
        """
        total_sessions = len(sessions)

        # 如果不指定数量，返回全部
        if num_sessions is None or num_sessions >= total_sessions:
            logger.info(f"  ✅ 使用全部 {total_sessions} 个sessions")
            return sessions

        # 验证数量
        if num_sessions <= 0:
            raise ValueError(f"num_sessions必须大于0，当前值: {num_sessions}")

        if selection_mode == 'first':
            selected = sessions[:num_sessions]
            logger.info(f"  ✅ 选择前 {num_sessions} 个sessions（共{total_sessions}个）")

        elif selection_mode == 'random':
            # 设置随机种子以支持可重复的随机选择
            if random_seed is not None:
                random.seed(random_seed)
                logger.info(f"  🎲 随机选择 {num_sessions} 个sessions（种子: {random_seed}）")
            else:
                logger.info(f"  🎲 随机选择 {num_sessions} 个sessions")

            selected = random.sample(sessions, num_sessions)

        else:
            raise ValueError(f"不支持的选择模式: {selection_mode}")

        # 输出选择的session信息
        selected_indices = []
        for sess in selected:
            # 找出原始索引
            for i, orig_sess in enumerate(sessions, 1):
                if orig_sess['session_id'] == sess['session_id']:
                    selected_indices.append(i)
                    break

        logger.info(f"  📋 选中的session编号: {sorted(selected_indices)}")

        return selected

    async def test_single_request(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        session_id: str = "",
        turn_number: int = 0
    ) -> Dict[str, Any]:
        """
        测试单个请求
        Returns:
            包含 success, duration_ms, input_tokens, output_tokens, ttft_ms, tps, error
        """
        start_time = time.perf_counter()

        try:
            if self.use_raw_httpx:
                result = await self._test_with_httpx_stream(
                    messages, system, max_tokens, temperature, start_time
                )
            elif self.api_format == "anthropic":
                result = await self._test_with_anthropic_stream(
                    messages, system, max_tokens, temperature, start_time
                )
            else:
                result = await self._test_with_openai_stream(
                    messages, system, max_tokens, temperature, start_time
                )

            duration_ms = (time.perf_counter() - start_time) * 1000

            return {
                'success': True,
                'duration_ms': duration_ms,
                'input_tokens': result.get('input_tokens', 0),
                'output_tokens': result.get('output_tokens', 0),
                'ttft_ms': result.get('ttft_ms'),
                'tps': result.get('tps'),
                'response_text': result.get('response_text', ''),  # 添加响应内容
                'error': None
            }

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            error_msg = str(e)
            logger.warning(f"  [{session_id}] Turn {turn_number} failed: {error_msg[:100]}")

            return {
                'success': False,
                'duration_ms': duration_ms,
                'input_tokens': 0,
                'output_tokens': 0,
                'ttft_ms': None,
                'tps': None,
                'error': error_msg
            }

    async def _test_with_httpx_stream(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        start_time: float
    ) -> Dict[str, Any]:
        """使用原生httpx测试（第三方API，支持流式）"""
        request_body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature if temperature is not None else 0.7,
            "stream": True  # 使用流式以测量TTFT和TPS
        }

        if system:
            request_body["system"] = system

        # 添加重复惩罚参数
        if self.repetition_penalty is not None:
            request_body["repetition_penalty"] = self.repetition_penalty

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        ttft_ms = None
        first_token_received = False
        generation_start = 0
        full_response = ""
        usage_data = None

        async with self.httpx_client.stream(
            "POST",
            self.api_url,
            json=request_body,
            headers=headers
        ) as response:
            response.raise_for_status()

            # 解析SSE流
            async for line in response.aiter_lines():
                if not line or not line.startswith('data: '):
                    continue

                data_str = line[6:]
                if data_str == '[DONE]':
                    break

                try:
                    event = json.loads(data_str)
                    event_type = event.get('type')

                    if not first_token_received and event_type == 'content_block_delta':
                        ttft_ms = (time.perf_counter() - start_time) * 1000
                        generation_start = time.perf_counter()
                        first_token_received = True

                    # 收集文本内容
                    if event_type == 'content_block_delta':
                        delta = event.get('delta', {})
                        if delta.get('type') == 'text_delta':
                            full_response += delta.get('text', '')

                    # 收集usage信息
                    elif event_type == 'message_delta':
                        usage = event.get('usage', {})
                        if usage:
                            usage_data = usage

                    elif event_type == 'message_start':
                        msg = event.get('message', {})
                        if msg.get('usage'):
                            usage_data = msg['usage']

                except json.JSONDecodeError:
                    continue

        # 计算TPS
        tps = None
        if first_token_received and usage_data:
            output_tokens = usage_data.get('output_tokens', 0)
            if output_tokens > 0:
                generation_time = time.perf_counter() - generation_start
                # 防止generation_time过小导致异常TPS值
                # 使用总时间计算TPS更稳定（从请求开始到结束）
                total_time = time.perf_counter() - start_time
                if total_time > 0:
                    tps = output_tokens / total_time

        return {
            'input_tokens': usage_data.get('input_tokens', 0) if usage_data else 0,
            'output_tokens': usage_data.get('output_tokens', 0) if usage_data else 0,
            'ttft_ms': ttft_ms,
            'tps': tps,
            'response_text': full_response  # 添加响应内容
        }

    async def _test_with_anthropic_stream(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        start_time: float
    ) -> Dict[str, Any]:
        """使用Anthropic SDK测试（流式，支持TTFT和TPS）"""
        request_params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature if temperature is not None else 0.7
        }

        if system:
            request_params["system"] = system

        ttft_ms = None
        first_token_received = False
        generation_start = 0
        full_response = ""
        usage_data = None

        async with self.client.messages.stream(**request_params) as stream:
            async for event in stream:
                # 检测第一个token
                if not first_token_received and hasattr(event, 'type'):
                    if event.type == 'content_block_delta':
                        ttft_ms = (time.perf_counter() - start_time) * 1000
                        generation_start = time.perf_counter()
                        first_token_received = True

                # 收集文本内容
                if hasattr(event, 'type') and event.type == 'content_block_delta':
                    if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                        full_response += event.delta.text

        # 获取最终消息以获取usage
        final_message = await stream.get_final_message()
        usage_data = final_message.usage

        # 计算TPS
        tps = None
        if first_token_received and usage_data.output_tokens > 0:
            generation_time = time.perf_counter() - generation_start
            # 防止generation_time过小导致异常TPS值
            # 使用总时间计算TPS更稳定（从请求开始到结束）
            total_time = time.perf_counter() - start_time
            if total_time > 0:
                tps = usage_data.output_tokens / total_time

        return {
            'input_tokens': usage_data.input_tokens,
            'output_tokens': usage_data.output_tokens,
            'ttft_ms': ttft_ms,
            'tps': tps,
            'response_text': full_response  # 添加响应内容
        }

    async def _test_with_openai_stream(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        start_time: float
    ) -> Dict[str, Any]:
        """使用OpenAI SDK测试（流式，支持TTFT和TPS）"""
        prepared_messages = []

        if system:
            prepared_messages.append({"role": "system", "content": system})

        prepared_messages.extend(messages)

        ttft_ms = None
        first_token_received = False
        generation_start = 0
        full_response = ""
        usage_data = None  # 用于存储流式返回的usage

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=prepared_messages,
            max_tokens=max_tokens or 4096,
            temperature=temperature if temperature is not None else 0.7,
            stream=True,
            stream_options={"include_usage": True},  # 请求返回usage信息
            # 添加重复惩罚参数
            **({"frequency_penalty": self.frequency_penalty} if self.frequency_penalty is not None else {}),
            **({"presence_penalty": self.presence_penalty} if self.presence_penalty is not None else {}),
            **({"extra_body": {"repetition_penalty": self.repetition_penalty}} if self.repetition_penalty is not None else {})
        )

        async for chunk in stream:
            # 检查并收集usage信息（在流的最后一个chunk中）
            if hasattr(chunk, 'usage') and chunk.usage is not None:
                usage_data = chunk.usage

            if not first_token_received and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    ttft_ms = (time.perf_counter() - start_time) * 1000
                    generation_start = time.perf_counter()
                    first_token_received = True

            # 收集文本内容
            if chunk.choices and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content

        # 优先使用API返回的usage，否则估算
        if usage_data:
            input_tokens = getattr(usage_data, 'prompt_tokens', 0)
            output_tokens = getattr(usage_data, 'completion_tokens', 0)
        else:
            # 回退到估算（兼容不支持stream_options的API）
            input_tokens = sum(len(str(m.get('content', '')).split()) for m in prepared_messages) * 1.3
            output_tokens = len(full_response.split()) * 1.3 if full_response else 0

        # 计算TPS
        tps = None
        if first_token_received and output_tokens > 0:
            # 使用总时间计算TPS更稳定（从请求开始到结束）
            total_time = time.perf_counter() - start_time
            if total_time > 0:
                tps = output_tokens / total_time

        return {
            'input_tokens': int(input_tokens),
            'output_tokens': int(output_tokens),
            'ttft_ms': ttft_ms,
            'tps': tps,
            'response_text': full_response
        }

    async def test_single_session(
        self,
        session_data: Dict[str, Any],
        session_index: int,
        max_turns: Optional[int] = None,
        rate_limit_delay: float = 0.0,
        warmup_turns: int = 0,
        cooldown_turns: int = 0,
        min_output_tokens: int = 0,
        skip_first_turns: int = 0,
        stop_event: Optional[asyncio.Event] = None
    ) -> ConcurrentSessionMetrics:
        """
        测试单个会话
        Args:
            session_data: 会话数据
            session_index: 会话编号
            max_turns: 最多测试多少轮
            rate_limit_delay: 每个请求之间的延迟（秒）
            warmup_turns: 排除前N轮的统计（预热阶段）
            cooldown_turns: 排除后N轮的统计（收尾阶段）
            min_output_tokens: 输出token数少于此值时不纳入统计（默认0表示全部纳入）
            skip_first_turns: 跳过每个session前N轮，不发起请求（默认0）
            stop_event: 停止事件，当设置时提前终止测试
        """
        session_id = session_data['session_id']
        title = session_data.get('title', f'Session {session_index}')
        turns_data = session_data['turns']

        # 先跳过前N轮（不发起请求）
        original_turn_count = len(turns_data)
        if skip_first_turns > 0:
            if skip_first_turns >= len(turns_data):
                logger.warning(f"⚠️  [{session_index}] skip_first_turns ({skip_first_turns}) >= 总轮数 ({len(turns_data)})，该session无可测试的轮次")
                # 返回一个空结果
                return ConcurrentSessionMetrics(
                    session_id=session_id,
                    session_index=session_index,
                    title=title,
                    total_turns=original_turn_count,
                    tested_turns=0,
                    start_time=datetime.now().isoformat(),
                    end_time=datetime.now().isoformat(),
                    total_duration_ms=0,
                    total_input_tokens=0,
                    total_output_tokens=0,
                    successful_turns=0,
                    failed_turns=0,
                    success_rate=0,
                    turn_details=[],
                    errors=[]
                )
            turns_data = turns_data[skip_first_turns:]

        # 再应用max_turns限制
        if max_turns:
            turns_data = turns_data[:max_turns]

        # 构建日志信息
        if skip_first_turns > 0:
            turn_range = f"第{skip_first_turns + 1}-{skip_first_turns + len(turns_data)}轮"
            logger.info(f"🔄 [{session_index}] 开始测试: {session_id[:16]}... ({turn_range}, 共{len(turns_data)}轮)")
        else:
            logger.info(f"🔄 [{session_index}] 开始测试: {session_id[:16]}... ({len(turns_data)} 轮)")

        start_time = datetime.now()
        total_input = 0
        total_output = 0
        successful = 0
        failed = 0
        errors = []
        all_durations = []
        all_ttft = []  # 收集TTFT数据
        all_tps = []   # 收集TPS数据
        turn_details = []  # 收集每轮详细数据（新增）

        # 用于统计的数据（排除warmup和cooldown）
        stable_durations = []
        stable_ttft = []
        stable_tps = []

        # 统计被排除的turns数量
        excluded_by_min_tokens = 0

        # 计算统计范围
        total_turns = len(turns_data)
        stats_start = warmup_turns  # 从第N轮开始统计
        stats_end = total_turns - cooldown_turns  # 到倒数第N轮结束

        for i, turn_data in enumerate(turns_data, 1):
            # 检查是否需要提前终止
            if stop_event and stop_event.is_set():
                logger.info(f"⏹️  [{session_index}] 收到停止信号，已完成 {i-1}/{len(turns_data)} 轮")
                break

            # 计算实际的turn编号（在原始session中的编号）
            actual_turn_number = skip_first_turns + i

            messages = turn_data['messages']
            system = turn_data.get('system')
            max_tokens = turn_data.get('max_tokens')
            temperature = turn_data.get('temperature')

            result = await self.test_single_request(
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                session_id=f"S{session_index}",
                turn_number=actual_turn_number
            )

            all_durations.append(result['duration_ms'])

            # 判断是否在稳定统计范围内（排除warmup和cooldown）
            turn_index = i - 1  # 转换为0-based索引
            is_stable_phase = stats_start <= turn_index < stats_end

            # 记录每轮详细数据（使用实际turn编号）
            turn_detail = {
                'turn_number': actual_turn_number,
                'success': result['success'],
                'duration_ms': result['duration_ms'],
                'input_tokens': result['input_tokens'],
                'output_tokens': result['output_tokens'],
                'ttft_ms': result.get('ttft_ms'),
                'tps': result.get('tps'),
                'response_text': result.get('response_text', ''),  # 添加响应内容
                'is_stable_phase': is_stable_phase,
                'error': result.get('error')
            }
            turn_details.append(turn_detail)

            if result['success']:
                successful += 1
                total_input += result['input_tokens']
                total_output += result['output_tokens']

                # 检查是否满足最小输出token要求
                output_tokens = result['output_tokens']
                meets_min_tokens = output_tokens >= min_output_tokens if min_output_tokens > 0 else True

                # 收集所有数据
                if result.get('ttft_ms') is not None:
                    all_ttft.append(result['ttft_ms'])
                if result.get('tps') is not None:
                    all_tps.append(result['tps'])

                # 只在稳定阶段且满足最小token要求时收集用于统计的数据
                if is_stable_phase:
                    if meets_min_tokens:
                        stable_durations.append(result['duration_ms'])
                        if result.get('ttft_ms') is not None:
                            stable_ttft.append(result['ttft_ms'])
                        if result.get('tps') is not None:
                            stable_tps.append(result['tps'])
                    else:
                        excluded_by_min_tokens += 1
            else:
                failed += 1
                errors.append(f"Turn {i}: {result['error']}")

            # 速率限制延迟
            if rate_limit_delay > 0 and i < len(turns_data):
                await asyncio.sleep(rate_limit_delay)

        end_time = datetime.now()
        total_duration = sum(all_durations)
        success_rate = (successful / len(turns_data) * 100) if turns_data else 0.0

        # 使用稳定阶段的数据计算统计值（如果有），否则使用全部数据
        ttft_data = stable_ttft if stable_ttft else all_ttft
        tps_data = stable_tps if stable_tps else all_tps

        avg_ttft = sum(ttft_data) / len(ttft_data) if ttft_data else None
        median_ttft = sorted(ttft_data)[len(ttft_data) // 2] if ttft_data else None
        avg_tps = sum(tps_data) / len(tps_data) if tps_data else None
        median_tps = sorted(tps_data)[len(tps_data) // 2] if tps_data else None

        # 日志输出
        if warmup_turns > 0 or cooldown_turns > 0 or min_output_tokens > 0:
            log_msg = f"✅ [{session_index}] 完成: 成功率 {success_rate:.1f}%, 耗时 {total_duration:.0f}ms "
            log_details = []

            if warmup_turns > 0 or cooldown_turns > 0:
                log_details.append(f"统计范围: 第{stats_start+1}-{stats_end}轮")

            if min_output_tokens > 0 and excluded_by_min_tokens > 0:
                log_details.append(f"排除<{min_output_tokens}tokens的turns: {excluded_by_min_tokens}个")

            if log_details:
                log_msg += f"({', '.join(log_details)}, 共{len(stable_ttft)}个有效样本)"
            else:
                log_msg += f"(共{len(stable_ttft)}个有效样本)"

            logger.info(log_msg)
        else:
            logger.info(f"✅ [{session_index}] 完成: 成功率 {success_rate:.1f}%, 耗时 {total_duration:.0f}ms")

        return ConcurrentSessionMetrics(
            session_id=session_id,
            session_index=session_index,
            title=title,
            total_turns=len(session_data['turns']),
            tested_turns=len(turns_data),
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_duration_ms=total_duration,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            successful_turns=successful,
            failed_turns=failed,
            success_rate=success_rate,
            avg_ttft_ms=avg_ttft,
            median_ttft_ms=median_ttft,
            avg_tps=avg_tps,
            median_tps=median_tps,
            valid_tps_samples=len(tps_data),
            valid_ttft_samples=len(ttft_data),
            turn_details=turn_details,  # 添加每轮详细数据
            errors=errors[:10]  # 只保留前10个错误
        )

    async def test_concurrent_sessions(
        self,
        sessions_data: List[Dict[str, Any]],
        max_concurrent: int = 3,
        max_turns_per_session: Optional[int] = None,
        rate_limit_delay: float = 0.5,
        warmup_turns: int = 0,
        cooldown_turns: int = 0,
        min_output_tokens: int = 0,
        skip_first_turns: int = 0,
        min_concurrent: Optional[int] = None
    ) -> ConcurrentTestReport:
        """
        并发测试多个会话
        Args:
            sessions_data: 会话数据列表
            max_concurrent: 最大并发会话数
            max_turns_per_session: 每个会话最多测试多少轮
            rate_limit_delay: 每个请求之间的延迟（秒），用于避免触发速率限制
            warmup_turns: 排除每个会话前N轮的统计（预热阶段）
            cooldown_turns: 排除每个会话后N轮的统计（收尾阶段）
            min_output_tokens: 输出token数少于此值时不纳入统计（默认0表示全部纳入）
            skip_first_turns: 跳过每个session前N轮，不发起请求（默认0）
            min_concurrent: 当剩余活跃会话数少于此值时停止测试
        """
        test_start = time.perf_counter()

        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 开始并发测试")
        logger.info(f"{'='*80}")
        logger.info(f"  总会话数: {len(sessions_data)}")
        logger.info(f"  最大并发数: {max_concurrent}")
        logger.info(f"  每会话最多测试: {max_turns_per_session or '全部'} 轮")
        if skip_first_turns > 0:
            logger.info(f"  跳过前N轮: {skip_first_turns} (不发起请求)")
        logger.info(f"  请求延迟: {rate_limit_delay}s")
        if warmup_turns > 0 or cooldown_turns > 0:
            logger.info(f"  统计范围: 排除前{warmup_turns}轮和后{cooldown_turns}轮")
        if min_output_tokens > 0:
            logger.info(f"  最小输出token数: {min_output_tokens} (少于此值不纳入统计)")
        if min_concurrent:
            logger.info(f"  最小并发数: {min_concurrent} (低于此值将停止测试)")
        logger.info(f"{'='*80}\n")

        # 创建停止事件（用于通知所有会话停止）
        stop_event = asyncio.Event()

        # 跟踪活跃会话数的原子计数器
        active_sessions = {'count': len(sessions_data)}
        active_lock = asyncio.Lock()

        # 创建信号量控制并发数
        semaphore = asyncio.Semaphore(max_concurrent)

        async def test_with_semaphore(session_data, index):
            async with semaphore:
                try:
                    result = await self.test_single_session(
                        session_data,
                        index,
                        max_turns_per_session,
                        rate_limit_delay,
                        warmup_turns,
                        cooldown_turns,
                        min_output_tokens,
                        skip_first_turns,
                        stop_event
                    )
                    return result
                finally:
                    # 会话完成，更新活跃数
                    async with active_lock:
                        active_sessions['count'] -= 1
                        remaining = active_sessions['count']

                        # 检查是否需要触发停止
                        if min_concurrent and remaining < min_concurrent and remaining > 0:
                            if not stop_event.is_set():
                                logger.warning(f"\n⚠️  剩余会话数 ({remaining}) 低于阈值 ({min_concurrent})，触发停止信号")
                                stop_event.set()

        # 创建所有任务
        tasks = [
            test_with_semaphore(session, i)
            for i, session in enumerate(sessions_data, 1)
        ]

        # 执行所有任务
        session_metrics = await asyncio.gather(*tasks)

        test_duration = (time.perf_counter() - test_start) * 1000

        # 汇总统计
        return self._generate_report(
            session_metrics,
            test_duration,
            max_concurrent,
            max_turns_per_session
        )

    def _generate_report(
        self,
        session_metrics: List[ConcurrentSessionMetrics],
        test_duration_ms: float,
        max_concurrent: int,
        max_turns_per_session: Optional[int]
    ) -> ConcurrentTestReport:
        """生成测试报告"""
        total_requests = sum(s.tested_turns for s in session_metrics)
        successful_requests = sum(s.successful_turns for s in session_metrics)
        failed_requests = sum(s.failed_turns for s in session_metrics)

        total_input_tokens = sum(s.total_input_tokens for s in session_metrics)
        total_output_tokens = sum(s.total_output_tokens for s in session_metrics)

        # 计算QPS
        qps = total_requests / (test_duration_ms / 1000) if test_duration_ms > 0 else 0

        # 计算响应时间（简化版，使用会话平均时间）
        all_avg_times = [s.total_duration_ms / s.tested_turns if s.tested_turns > 0 else 0
                        for s in session_metrics]
        avg_response_time = sum(all_avg_times) / len(all_avg_times) if all_avg_times else 0

        sorted_times = sorted(all_avg_times)
        p50_idx = int(len(sorted_times) * 0.5)
        p95_idx = int(len(sorted_times) * 0.95)
        p99_idx = int(len(sorted_times) * 0.99)

        p50 = sorted_times[p50_idx] if sorted_times else 0
        p95 = sorted_times[p95_idx] if sorted_times else 0
        p99 = sorted_times[p99_idx] if sorted_times else 0

        # 错误汇总
        total_errors = sum(len(s.errors) for s in session_metrics)
        error_types = {}
        for s in session_metrics:
            for error in s.errors:
                error_type = error.split(':')[0] if ':' in error else 'Unknown'
                error_types[error_type] = error_types.get(error_type, 0) + 1

        # 汇总TTFT和TPS指标
        all_session_ttft = [s.avg_ttft_ms for s in session_metrics if s.avg_ttft_ms is not None]
        all_session_tps = [s.avg_tps for s in session_metrics if s.avg_tps is not None]

        # 计算全局TTFT统计（简单平均，因为TTFT不受样本数影响）
        avg_ttft = sum(all_session_ttft) / len(all_session_ttft) if all_session_ttft else None
        median_ttft = sorted(all_session_ttft)[len(all_session_ttft) // 2] if all_session_ttft else None
        p95_ttft_idx = int(len(all_session_ttft) * 0.95)
        p95_ttft = sorted(all_session_ttft)[p95_ttft_idx] if all_session_ttft else None

        # 计算全局TPS统计（使用加权平均）
        # 使用每个会话的有效样本数作为权重
        sessions_with_tps = [s for s in session_metrics if s.avg_tps is not None and s.valid_tps_samples > 0]

        if sessions_with_tps:
            # 加权平均TPS
            total_weighted_tps = sum(s.avg_tps * s.valid_tps_samples for s in sessions_with_tps)
            total_samples = sum(s.valid_tps_samples for s in sessions_with_tps)
            avg_tps = total_weighted_tps / total_samples if total_samples > 0 else None

            # 中位数：将所有会话的TPS样本展平后计算（近似，使用会话平均TPS）
            # 注意：这是简化版本，理想情况下应该收集所有单个请求的TPS
            median_tps = sorted(all_session_tps)[len(all_session_tps) // 2] if all_session_tps else None
        else:
            avg_tps = None
            median_tps = None

        return ConcurrentTestReport(
            provider_name=self.provider_name,
            model_name=self.model,
            api_url=self.api_url,
            test_time=datetime.now().isoformat(),
            total_sessions=len(session_metrics),
            max_concurrent_sessions=max_concurrent,
            max_turns_per_session=max_turns_per_session,
            total_test_duration_ms=test_duration_ms,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            overall_success_rate=(successful_requests / total_requests * 100) if total_requests > 0 else 0,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
            requests_per_second=qps,
            avg_response_time_ms=avg_response_time,
            p50_response_time_ms=p50,
            p95_response_time_ms=p95,
            p99_response_time_ms=p99,
            avg_ttft_ms=avg_ttft,
            median_ttft_ms=median_ttft,
            p95_ttft_ms=p95_ttft,
            avg_tps=avg_tps,
            median_tps=median_tps,
            sessions=session_metrics,
            total_errors=total_errors,
            error_types=error_types
        )

    def print_report(self, report: ConcurrentTestReport):
        """打印测试报告"""
        print("\n" + "="*80)
        print("📊 并发测试报告")
        print("="*80)

        print(f"\n🎯 测试配置:")
        print(f"  提供商: {report.provider_name}")
        print(f"  模型: {report.model_name}")
        print(f"  API URL: {report.api_url}")
        print(f"  测试时间: {report.test_time}")

        print(f"\n⚙️  并发配置:")
        print(f"  总会话数: {report.total_sessions}")
        print(f"  最大并发数: {report.max_concurrent_sessions}")
        print(f"  每会话轮数: {report.max_turns_per_session or '全部'}")

        print(f"\n📈 总体统计:")
        print(f"  总测试时长: {report.total_test_duration_ms / 1000:.2f}s")
        print(f"  总请求数: {report.total_requests}")
        print(f"  成功请求: {report.successful_requests}")
        print(f"  失败请求: {report.failed_requests}")
        print(f"  成功率: {report.overall_success_rate:.1f}%")

        print(f"\n⚡ 性能指标:")
        print(f"  QPS (请求/秒): {report.requests_per_second:.2f}")
        print(f"  平均响应时间: {report.avg_response_time_ms:.0f}ms")
        print(f"  P50 响应时间: {report.p50_response_time_ms:.0f}ms")
        print(f"  P95 响应时间: {report.p95_response_time_ms:.0f}ms")
        print(f"  P99 响应时间: {report.p99_response_time_ms:.0f}ms")

        # 显示TTFT和TPS指标
        if report.avg_ttft_ms is not None or report.avg_tps is not None:
            print(f"\n🚀 TTFT和TPS指标:")
            if report.avg_ttft_ms is not None:
                print(f"  平均TTFT: {report.avg_ttft_ms:.0f}ms")
                if report.median_ttft_ms is not None:
                    print(f"  中位TTFT: {report.median_ttft_ms:.0f}ms")
                if report.p95_ttft_ms is not None:
                    print(f"  P95 TTFT: {report.p95_ttft_ms:.0f}ms")
            if report.avg_tps is not None:
                print(f"  平均TPS: {report.avg_tps:.2f} tokens/s")
                if report.median_tps is not None:
                    print(f"  中位TPS: {report.median_tps:.2f} tokens/s")

        print(f"\n🎯 Token统计:")
        print(f"  输入Token: {report.total_input_tokens:,}")
        print(f"  输出Token: {report.total_output_tokens:,}")
        print(f"  总Token: {report.total_tokens:,}")

        if report.total_errors > 0:
            print(f"\n⚠️  错误统计:")
            print(f"  总错误数: {report.total_errors}")
            print(f"  错误类型:")
            for error_type, count in sorted(report.error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"    {error_type}: {count}")

        print("\n" + "="*80 + "\n")

    def save_report(self, report: ConcurrentTestReport, output_file: str):
        """保存测试报告"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)

        logger.info(f"💾 报告已保存: {output_path}")

    def generate_tps_distribution_chart(
        self,
        report: ConcurrentTestReport,
        output_dir: str,
        chart_format: str = 'both',
        show_content_threshold: int = 100
    ):
        """
        生成TPS分布曲线图表
        Args:
            report: 测试报告
            output_dir: 输出目录
            chart_format: 图表格式 ('matplotlib', 'plotly', 'both')
            show_content_threshold: 输出token数小于此值时显示内容（0表示不显示）
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 为每个session生成TPS曲线
        if chart_format in ['matplotlib', 'both'] and MATPLOTLIB_AVAILABLE:
            self._generate_matplotlib_charts(report, output_path, show_content_threshold)

        if chart_format in ['plotly', 'both'] and PLOTLY_AVAILABLE:
            self._generate_plotly_charts(report, output_path, show_content_threshold)

    def _generate_matplotlib_charts(self, report: ConcurrentTestReport, output_path: Path, show_content_threshold: int = 100):
        """使用matplotlib生成TPS分布曲线"""
        logger.info("📊 生成matplotlib图表...")

        # 为所有session生成一个大图
        num_sessions = len(report.sessions)
        fig, axes = plt.subplots(
            num_sessions, 1,
            figsize=(14, 4 * num_sessions),
            squeeze=False
        )

        for idx, session in enumerate(report.sessions):
            ax = axes[idx, 0]

            # 提取完整的turn数据
            turn_data = {'stable': [], 'warmup': []}

            for turn_detail in session.turn_details:
                if turn_detail['success'] and turn_detail['tps'] is not None:
                    data_point = {
                        'turn': turn_detail['turn_number'],
                        'tps': turn_detail['tps'],
                        'ttft': turn_detail.get('ttft_ms', 0),
                        'output_tokens': turn_detail.get('output_tokens', 0),
                    }

                    if turn_detail['is_stable_phase']:
                        turn_data['stable'].append(data_point)
                    else:
                        turn_data['warmup'].append(data_point)

            has_data = len(turn_data['stable']) > 0 or len(turn_data['warmup']) > 0

            if has_data:
                # 绘制稳定阶段的数据
                if turn_data['stable']:
                    stable_points = turn_data['stable']
                    ax.plot([p['turn'] for p in stable_points],
                           [p['tps'] for p in stable_points],
                           'o-', color='#2E86AB', linewidth=2, markersize=6,
                           label='Stable Phase', alpha=0.8)

                # 绘制预热/收尾阶段的数据
                if turn_data['warmup']:
                    warmup_points = turn_data['warmup']
                    ax.plot([p['turn'] for p in warmup_points],
                           [p['tps'] for p in warmup_points],
                           'o--', color='#A23B72', linewidth=1.5, markersize=4,
                           label='Warmup/Cooldown', alpha=0.6)

                # 添加平均线
                if session.avg_tps:
                    ax.axhline(y=session.avg_tps, color='#F18F01',
                              linestyle='--', linewidth=2,
                              label=f'Avg TPS: {session.avg_tps:.2f}', alpha=0.8)

                # 计算统计信息
                all_points = turn_data['stable'] + turn_data['warmup']
                avg_ttft = sum(p['ttft'] for p in all_points) / len(all_points)
                avg_output_tokens = sum(p['output_tokens'] for p in all_points) / len(all_points)

                # 设置标题和统计信息
                title = f'Session {session.session_index}: {session.title[:50]}\n'
                title += f'Avg TTFT: {avg_ttft:.1f}ms | Avg Output: {avg_output_tokens:.0f} tokens'
                ax.set_title(title, fontsize=10, fontweight='bold', pad=10)

                ax.set_xlabel('Turn Number', fontsize=10)
                ax.set_ylabel('TPS (tokens/s)', fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(loc='best', fontsize=9)

                # 设置y轴范围（避免过小的波动）
                all_tps = [p['tps'] for p in all_points]
                if all_tps:
                    y_min = min(all_tps) * 0.9
                    y_max = max(all_tps) * 1.1
                    ax.set_ylim([y_min, y_max])
            else:
                ax.text(0.5, 0.5, 'No TPS data available',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, color='gray')
                ax.set_xlabel('Turn Number')
                ax.set_ylabel('TPS (tokens/s)')

        plt.suptitle(
            f'TPS Distribution - {report.provider_name} ({report.model_name})',
            fontsize=14, fontweight='bold', y=0.995
        )
        plt.tight_layout()

        # 保存图表
        chart_file = output_path / 'tps_distribution_matplotlib.png'
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  ✅ Matplotlib图表已保存: {chart_file}")

    def _generate_plotly_charts(self, report: ConcurrentTestReport, output_path: Path, show_content_threshold: int = 100):
        """使用plotly生成TPS分布曲线（交互式）"""
        logger.info("📊 生成Plotly交互式图表...")

        # 创建子图
        num_sessions = len(report.sessions)
        fig = make_subplots(
            rows=num_sessions, cols=1,
            subplot_titles=[f'Session {s.session_index}: {s.title[:50]}'
                          for s in report.sessions],
            vertical_spacing=0.08 / num_sessions if num_sessions > 1 else 0.1
        )

        for idx, session in enumerate(report.sessions, 1):
            # 提取完整的turn数据
            turn_data = {'stable': [], 'warmup': []}

            for turn_detail in session.turn_details:
                if turn_detail['success'] and turn_detail['tps'] is not None:
                    # 截断并转义响应内容
                    response_text = turn_detail.get('response_text', '')
                    output_tokens = turn_detail.get('output_tokens', 0)

                    # 如果满足阈值条件且有内容，则截断显示
                    display_text = ''
                    if show_content_threshold > 0 and output_tokens < show_content_threshold and response_text:
                        # 去除前后空白
                        response_text = response_text.strip()
                        if response_text:  # 确保不是空字符串
                            # 截断到最多300字符，避免hover框过大
                            display_text = response_text[:300]
                            if len(response_text) > 300:
                                display_text += '...'
                            # HTML转义特殊字符
                            display_text = display_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                            display_text = display_text.replace('\n', '<br>')  # 换行转为HTML
                            # 添加分隔线和标题
                            display_text = '<br>---<br><b>Response:</b><br>' + display_text

                    data_point = {
                        'turn': turn_detail['turn_number'],
                        'tps': turn_detail['tps'],
                        'ttft': turn_detail.get('ttft_ms', 0),
                        'output_tokens': output_tokens,
                        'input_tokens': turn_detail.get('input_tokens', 0),
                        'response_text': display_text,
                    }

                    if turn_detail['is_stable_phase']:
                        turn_data['stable'].append(data_point)
                    else:
                        turn_data['warmup'].append(data_point)

            has_data = len(turn_data['stable']) > 0 or len(turn_data['warmup']) > 0

            if has_data:
                # 添加稳定阶段的曲线
                if turn_data['stable']:
                    stable_points = turn_data['stable']
                    fig.add_trace(
                        go.Scatter(
                            x=[p['turn'] for p in stable_points],
                            y=[p['tps'] for p in stable_points],
                            mode='lines+markers',
                            name=f'S{session.session_index} Stable',
                            line=dict(color='#2E86AB', width=2),
                            marker=dict(size=6),
                            customdata=[[p['ttft'], p['output_tokens'], p['input_tokens'], p['response_text']]
                                       for p in stable_points],
                            hovertemplate=(
                                '<b>Turn %{x}</b><br>'
                                'TPS: %{y:.2f} tokens/s<br>'
                                'TTFT: %{customdata[0]:.1f} ms<br>'
                                'Output Tokens: %{customdata[1]}<br>'
                                'Input Tokens: %{customdata[2]}<br>'
                                '%{customdata[3]}'  # 响应内容（如果有）
                                '<extra></extra>'
                            )
                        ),
                        row=idx, col=1
                    )

                # 添加预热/收尾阶段的曲线
                if turn_data['warmup']:
                    warmup_points = turn_data['warmup']
                    fig.add_trace(
                        go.Scatter(
                            x=[p['turn'] for p in warmup_points],
                            y=[p['tps'] for p in warmup_points],
                            mode='lines+markers',
                            name=f'S{session.session_index} Warmup/Cooldown',
                            line=dict(color='#A23B72', width=1.5, dash='dash'),
                            marker=dict(size=4),
                            customdata=[[p['ttft'], p['output_tokens'], p['input_tokens'], p['response_text']]
                                       for p in warmup_points],
                            hovertemplate=(
                                '<b>Turn %{x}</b><br>'
                                'TPS: %{y:.2f} tokens/s<br>'
                                'TTFT: %{customdata[0]:.1f} ms<br>'
                                'Output Tokens: %{customdata[1]}<br>'
                                'Input Tokens: %{customdata[2]}<br>'
                                '%{customdata[3]}'  # 响应内容（如果有）
                                '<extra></extra>'
                            )
                        ),
                        row=idx, col=1
                    )

                # 添加平均线
                if session.avg_tps:
                    all_turns = [p['turn'] for p in turn_data['stable'] + turn_data['warmup']]
                    if all_turns:
                        fig.add_trace(
                            go.Scatter(
                                x=[min(all_turns), max(all_turns)],
                                y=[session.avg_tps, session.avg_tps],
                                mode='lines',
                                name=f'S{session.session_index} Avg: {session.avg_tps:.2f}',
                                line=dict(color='#F18F01', width=2, dash='dash'),
                                hovertemplate='Avg TPS: %{y:.2f}<extra></extra>'
                            ),
                            row=idx, col=1
                        )

            # 更新坐标轴
            fig.update_xaxes(title_text='Turn Number', row=idx, col=1)
            fig.update_yaxes(title_text='TPS (tokens/s)', row=idx, col=1)

        # 更新布局
        fig.update_layout(
            title_text=f'TPS Distribution - {report.provider_name} ({report.model_name})',
            height=400 * num_sessions,
            showlegend=True,
            hovermode='closest',  # 改用closest模式，更适合显示长内容
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="monospace",
                align="left",
                namelength=-1  # 不截断label
            )
        )

        # 保存HTML文件，添加自定义CSS来控制hover框大小
        html_file = output_path / 'tps_distribution_plotly.html'

        # 生成HTML并添加自定义CSS
        html_string = fig.to_html(include_plotlyjs='cdn')

        # 插入自定义CSS来限制hover框大小并添加滚动
        custom_css = """
<style>
.hoverlayer .hovertext {
    max-width: 600px !important;
    max-height: 400px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    word-wrap: break-word !important;
    white-space: pre-wrap !important;
}
</style>
"""
        # 在</head>标签前插入CSS
        html_string = html_string.replace('</head>', custom_css + '</head>')

        # 写入文件
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_string)

        logger.info(f"  ✅ Plotly图表已保存: {html_file}")


async def main():
    parser = argparse.ArgumentParser(
        description="并发多会话回放测试工具"
    )
    parser.add_argument(
        '--input',
        required=True,
        help='输入的多会话JSON文件'
    )
    parser.add_argument(
        '--num-sessions',
        type=int,
        help='要测试的会话数量（默认使用全部）'
    )
    parser.add_argument(
        '--selection-mode',
        choices=['first', 'random'],
        default='first',
        help='会话选择模式: first=前N个, random=随机N个（默认first）'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        help='随机选择的种子（用于可重复的随机选择）'
    )
    parser.add_argument(
        '--api-url',
        required=True,
        help='API URL'
    )
    parser.add_argument(
        '--api-key',
        required=True,
        help='API Key'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='模型名称'
    )
    parser.add_argument(
        '--provider',
        default='Test Provider',
        help='供应商名称'
    )
    parser.add_argument(
        '--api-format',
        choices=['anthropic', 'openai'],
        default='anthropic',
        help='API格式'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=3,
        help='最大并发会话数（默认3）'
    )
    parser.add_argument(
        '--max-turns',
        type=int,
        help='每个会话最多测试多少轮'
    )
    parser.add_argument(
        '--rate-limit-delay',
        type=float,
        default=0.5,
        help='每个请求之间的延迟（秒，默认0.5）'
    )
    parser.add_argument(
        '--warmup-turns',
        type=int,
        default=0,
        help='排除每个会话前N轮的统计（预热阶段，默认0）'
    )
    parser.add_argument(
        '--cooldown-turns',
        type=int,
        default=0,
        help='排除每个会话后N轮的统计（收尾阶段，默认0）'
    )
    parser.add_argument(
        '--min-concurrent',
        type=int,
        help='当剩余活跃会话数少于此值时停止测试（可选，避免低并发导致TPS异常）'
    )
    parser.add_argument(
        '--min-output-tokens',
        type=int,
        default=16,
        help='输出token数少于此值时不纳入统计（默认16，0表示全部纳入）'
    )
    parser.add_argument(
        '--skip-first-turns',
        type=int,
        default=0,
        help='跳过每个session前N轮，不发起请求（默认0）'
    )
    parser.add_argument(
        '--output',
        help='输出报告文件路径'
    )
    parser.add_argument(
        '--generate-charts',
        action='store_true',
        help='生成TPS分布曲线图表'
    )
    parser.add_argument(
        '--chart-format',
        choices=['matplotlib', 'plotly', 'both'],
        default='both',
        help='图表格式（默认both）'
    )
    parser.add_argument(
        '--show-content-threshold',
        type=int,
        default=100,
        help='当输出token数小于此阈值时，在图表中显示响应内容（默认100，0表示不显示）'
    )
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        help='重复惩罚参数（适用于vLLM等，通常 > 1.0）'
    )
    parser.add_argument(
        '--frequency-penalty',
        type=float,
        help='频率惩罚参数（OpenAI标准，范围 -2.0 到 2.0）'
    )
    parser.add_argument(
        '--presence-penalty',
        type=float,
        help='存在惩罚参数（OpenAI标准，范围 -2.0 到 2.0）'
    )

    args = parser.parse_args()

    # 创建测试器
    tester = ConcurrentTester(
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        provider_name=args.provider,
        api_format=args.api_format,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty
    )

    # 加载会话数据
    data = tester.load_sessions_data(args.input)
    sessions = data.get('sessions', [])

    if not sessions:
        logger.error("❌ 没有找到会话数据")
        return 1

    # 选择要测试的sessions
    sessions = tester.select_sessions(
        sessions=sessions,
        num_sessions=args.num_sessions,
        selection_mode=args.selection_mode,
        random_seed=args.random_seed
    )

    # 执行并发测试
    report = await tester.test_concurrent_sessions(
        sessions_data=sessions,
        max_concurrent=args.max_concurrent,
        max_turns_per_session=args.max_turns,
        rate_limit_delay=args.rate_limit_delay,
        warmup_turns=args.warmup_turns,
        cooldown_turns=args.cooldown_turns,
        min_output_tokens=args.min_output_tokens,
        skip_first_turns=args.skip_first_turns,
        min_concurrent=args.min_concurrent
    )

    # 打印报告
    tester.print_report(report)

    # 确定输出目录
    if args.output:
        output_file = args.output
        output_dir = str(Path(output_file).parent)
    else:
        # 默认输出文件名
        provider = args.provider.replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = "benchmark_results"
        output_file = f"{output_dir}/concurrent_test_{provider}_{timestamp}.json"

    # 保存报告
    tester.save_report(report, output_file)

    # 生成TPS分布曲线图表
    if args.generate_charts:
        logger.info("\n" + "="*80)
        logger.info("📊 生成TPS分布曲线图表")
        if args.show_content_threshold > 0:
            logger.info(f"  📝 将显示输出token数 < {args.show_content_threshold} 的响应内容")
        logger.info("="*80)
        tester.generate_tps_distribution_chart(
            report=report,
            output_dir=output_dir,
            chart_format=args.chart_format,
            show_content_threshold=args.show_content_threshold
        )
        logger.info("="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))