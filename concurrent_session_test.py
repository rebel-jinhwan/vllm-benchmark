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

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 로깅 설정
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
    logger.error(f"필수 의존성 라이브러리 누락: {e}")
    logger.error("설치: pip install openai anthropic httpx")
    sys.exit(1)

# 시각화 라이브러리 (선택)
try:
    import matplotlib
    matplotlib.use('Agg')  # 비대화형 백엔드 사용
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib이 설치되지 않아 matplotlib 차트를 생성할 수 없습니다")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly가 설치되지 않아 plotly 차트를 생성할 수 없습니다")

# 랜덤 선택용 random 임포트
import random


@dataclass
class ConcurrentSessionMetrics:
    """동시 세션 성능 지표"""
    session_id: str
    session_index: int  # 세션 번호 (식별용)
    title: str
    total_turns: int
    tested_turns: int  # 실제 테스트한 턴 수

    # 시간 지표
    start_time: str
    end_time: str
    total_duration_ms: float

    # 토큰 지표
    total_input_tokens: int
    total_output_tokens: int

    # 성공률
    successful_turns: int
    failed_turns: int
    success_rate: float

    # TTFT 및 TPS 지표 (추가됨)
    avg_ttft_ms: Optional[float] = None
    median_ttft_ms: Optional[float] = None
    avg_tps: Optional[float] = None
    median_tps: Optional[float] = None

    # 유효 샘플 수 (가중 평균용)
    valid_tps_samples: int = 0
    valid_ttft_samples: int = 0

    # 턴별 상세 데이터 (추가됨)
    turn_details: List[Dict[str, Any]] = field(default_factory=list)

    # 오류 정보
    errors: List[str] = field(default_factory=list)


@dataclass
class ConcurrentTestReport:
    """동시성 테스트 보고서"""
    provider_name: str
    model_name: str
    api_url: str
    test_time: str

    # 동시성 설정
    total_sessions: int
    max_concurrent_sessions: int
    max_turns_per_session: Optional[int]

    # 전체 통계
    total_test_duration_ms: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    overall_success_rate: float

    # 토큰 통계
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int

    # 성능 지표
    requests_per_second: float  # QPS
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float

    # 세션별 결과
    sessions: List[ConcurrentSessionMetrics]

    # 오류 요약
    total_errors: int

    # TTFT 및 TPS 지표 (추가됨)
    avg_ttft_ms: Optional[float] = None
    median_ttft_ms: Optional[float] = None
    p95_ttft_ms: Optional[float] = None
    avg_tps: Optional[float] = None
    median_tps: Optional[float] = None
    error_types: Dict[str, int] = field(default_factory=dict)


class ConcurrentTester:
    """동시성 테스터"""

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
        """테스터 초기화"""
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.provider_name = provider_name
        self.api_format = api_format
        self.use_raw_httpx = False
        # 반복 패널티 파라미터
        self.repetition_penalty = repetition_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        # 클라이언트 초기화
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
                logger.info(f"  ⚙️  네이티브 httpx 클라이언트 사용 (서드파티 API)")
            else:
                self.client = anthropic.AsyncAnthropic(
                    api_key=api_key,
                    base_url=base_url
                )
        else:
            # OpenAI SDK가 자동으로 /chat/completions를 추가하므로 제거 필요
            base_url = api_url
            if base_url.endswith('/chat/completions'):
                base_url = base_url.rsplit('/chat/completions', 1)[0]

            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )

    def load_sessions_data(self, json_file: str) -> Dict[str, Any]:
        """다중 세션 데이터 로드"""
        path = Path(json_file)
        if not path.exists():
            raise FileNotFoundError(f"파일이 존재하지 않습니다: {json_file}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"📂 데이터 로드: {json_file}")
        logger.info(f"  세션 수: {data.get('total_sessions', len(data.get('sessions', [])))}")

        return data

    def select_sessions(
        self,
        sessions: List[Dict[str, Any]],
        num_sessions: Optional[int] = None,
        selection_mode: str = 'first',
        random_seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        테스트할 세션 선택
        Args:
            sessions: 전체 세션 목록
            num_sessions: 선택할 세션 수 (None이면 전체)
            selection_mode: 선택 모드 ('first': 처음 N개, 'random': 랜덤 N개)
            random_seed: 랜덤 시드 (재현 가능한 랜덤 선택용)
        Returns:
            선택된 세션 목록
        """
        total_sessions = len(sessions)

        # 수량을 지정하지 않으면 전체 반환
        if num_sessions is None or num_sessions >= total_sessions:
            logger.info(f"  ✅ 전체 {total_sessions}개 세션 사용")
            return sessions

        # 수량 검증
        if num_sessions <= 0:
            raise ValueError(f"num_sessions는 0보다 커야 합니다. 현재 값: {num_sessions}")

        if selection_mode == 'first':
            selected = sessions[:num_sessions]
            logger.info(f"  ✅ 처음 {num_sessions}개 세션 선택 (총 {total_sessions}개)")

        elif selection_mode == 'random':
            # 재현 가능한 랜덤 선택을 위해 랜덤 시드 설정
            if random_seed is not None:
                random.seed(random_seed)
                logger.info(f"  🎲 랜덤으로 {num_sessions}개 세션 선택 (시드: {random_seed})")
            else:
                logger.info(f"  🎲 랜덤으로 {num_sessions}개 세션 선택")

            selected = random.sample(sessions, num_sessions)

        else:
            raise ValueError(f"지원하지 않는 선택 모드: {selection_mode}")

        # 선택된 세션 정보 출력
        selected_indices = []
        for sess in selected:
            # 원본 인덱스 찾기
            for i, orig_sess in enumerate(sessions, 1):
                if orig_sess['session_id'] == sess['session_id']:
                    selected_indices.append(i)
                    break

        logger.info(f"  📋 선택된 세션 번호: {sorted(selected_indices)}")

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
        단일 요청 테스트
        Returns:
            success, duration_ms, input_tokens, output_tokens, ttft_ms, tps, error 포함
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
                'response_text': result.get('response_text', ''),  # 응답 내용 추가
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
        """네이티브 httpx로 테스트 (서드파티 API, 스트리밍 지원)"""
        request_body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature if temperature is not None else 0.7,
            "stream": True  # TTFT 및 TPS 측정을 위해 스트리밍 사용
        }

        if system:
            request_body["system"] = system

        # 반복 패널티 파라미터 추가
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

            # SSE 스트림 파싱
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

                    # 텍스트 내용 수집
                    if event_type == 'content_block_delta':
                        delta = event.get('delta', {})
                        if delta.get('type') == 'text_delta':
                            full_response += delta.get('text', '')

                    # usage 정보 수집
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

        # TPS 계산
        tps = None
        if first_token_received and usage_data:
            output_tokens = usage_data.get('output_tokens', 0)
            if output_tokens > 0:
                generation_time = time.perf_counter() - generation_start
                # generation_time이 너무 작아 비정상 TPS가 나오는 것 방지
                # 총 시간으로 TPS 계산이 더 안정적 (요청 시작~종료)
                total_time = time.perf_counter() - start_time
                if total_time > 0:
                    tps = output_tokens / total_time

        return {
            'input_tokens': usage_data.get('input_tokens', 0) if usage_data else 0,
            'output_tokens': usage_data.get('output_tokens', 0) if usage_data else 0,
            'ttft_ms': ttft_ms,
            'tps': tps,
            'response_text': full_response  # 응답 내용 추가
        }

    async def _test_with_anthropic_stream(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        start_time: float
    ) -> Dict[str, Any]:
        """Anthropic SDK로 테스트 (스트리밍, TTFT/TPS 지원)"""
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
                # 첫 토큰 감지
                if not first_token_received and hasattr(event, 'type'):
                    if event.type == 'content_block_delta':
                        ttft_ms = (time.perf_counter() - start_time) * 1000
                        generation_start = time.perf_counter()
                        first_token_received = True

                # 텍스트 내용 수집
                if hasattr(event, 'type') and event.type == 'content_block_delta':
                    if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                        full_response += event.delta.text

        # usage를 위해 최종 메시지 가져오기
        final_message = await stream.get_final_message()
        usage_data = final_message.usage

        # TPS 계산
        tps = None
        if first_token_received and usage_data.output_tokens > 0:
            generation_time = time.perf_counter() - generation_start
            # generation_time이 너무 작아 비정상 TPS가 나오는 것 방지
            # 총 시간으로 TPS 계산이 더 안정적 (요청 시작~종료)
            total_time = time.perf_counter() - start_time
            if total_time > 0:
                tps = usage_data.output_tokens / total_time

        return {
            'input_tokens': usage_data.input_tokens,
            'output_tokens': usage_data.output_tokens,
            'ttft_ms': ttft_ms,
            'tps': tps,
            'response_text': full_response  # 응답 내용 추가
        }

    async def _test_with_openai_stream(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        start_time: float
    ) -> Dict[str, Any]:
        """OpenAI SDK로 테스트 (스트리밍, TTFT/TPS 지원)"""
        prepared_messages = []

        if system:
            prepared_messages.append({"role": "system", "content": system})

        prepared_messages.extend(messages)

        ttft_ms = None
        first_token_received = False
        generation_start = 0
        full_response = ""
        usage_data = None  # 스트리밍 반환 usage 저장용

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=prepared_messages,
            max_tokens=max_tokens or 4096,
            temperature=temperature if temperature is not None else 0.7,
            stream=True,
            stream_options={"include_usage": True},  # 요청 시 usage 정보 반환
            # 반복 패널티 파라미터 추가
            **({"frequency_penalty": self.frequency_penalty} if self.frequency_penalty is not None else {}),
            **({"presence_penalty": self.presence_penalty} if self.presence_penalty is not None else {}),
            **({"extra_body": {"repetition_penalty": self.repetition_penalty}} if self.repetition_penalty is not None else {})
        )

        async for chunk in stream:
            # usage 정보 확인 및 수집 (스트림 마지막 chunk에서)
            if hasattr(chunk, 'usage') and chunk.usage is not None:
                usage_data = chunk.usage

            if not first_token_received and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    ttft_ms = (time.perf_counter() - start_time) * 1000
                    generation_start = time.perf_counter()
                    first_token_received = True

            # 텍스트 내용 수집
            if chunk.choices and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content

        # API 반환 usage 우선 사용, 없으면 추정
        if usage_data:
            input_tokens = getattr(usage_data, 'prompt_tokens', 0)
            output_tokens = getattr(usage_data, 'completion_tokens', 0)
        else:
            # 추정으로 폴백 (stream_options 미지원 API 호환)
            input_tokens = sum(len(str(m.get('content', '')).split()) for m in prepared_messages) * 1.3
            output_tokens = len(full_response.split()) * 1.3 if full_response else 0

        # TPS 계산
        tps = None
        if first_token_received and output_tokens > 0:
            # 총 시간으로 TPS 계산이 더 안정적 (요청 시작~종료)
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
        단일 세션 테스트
        Args:
            session_data: 세션 데이터
            session_index: 세션 번호
            max_turns: 최대 테스트 턴 수
            rate_limit_delay: 요청 간 지연 (초)
            warmup_turns: 앞 N턴 통계 제외 (웜업 단계)
            cooldown_turns: 뒤 N턴 통계 제외 (쿨다운 단계)
            min_output_tokens: 출력 토큰이 이 값보다 적으면 통계 제외 (기본 0=전체 포함)
            skip_first_turns: 각 세션 앞 N턴 건너뛰기, 요청 안 함 (기본 0)
            stop_event: 중지 이벤트, 설정 시 테스트 조기 종료
        """
        session_id = session_data['session_id']
        title = session_data.get('title', f'Session {session_index}')
        turns_data = session_data['turns']

        # 먼저 앞 N턴 건너뛰기 (요청 안 함)
        original_turn_count = len(turns_data)
        if skip_first_turns > 0:
            if skip_first_turns >= len(turns_data):
                logger.warning(f"⚠️  [{session_index}] skip_first_turns ({skip_first_turns}) >= 총 턴 수 ({len(turns_data)}), 해당 세션에 테스트할 턴이 없음")
                # 빈 결과 반환
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

        # max_turns 제한 적용
        if max_turns:
            turns_data = turns_data[:max_turns]

        # 로그 메시지 구성
        if skip_first_turns > 0:
            turn_range = f"제{skip_first_turns + 1}-{skip_first_turns + len(turns_data)}턴"
            logger.info(f"🔄 [{session_index}] 테스트 시작: {session_id[:16]}... ({turn_range}, 총 {len(turns_data)}턴)")
        else:
            logger.info(f"🔄 [{session_index}] 테스트 시작: {session_id[:16]}... ({len(turns_data)}턴)")

        start_time = datetime.now()
        total_input = 0
        total_output = 0
        successful = 0
        failed = 0
        errors = []
        all_durations = []
        all_ttft = []  # TTFT 데이터 수집
        all_tps = []   # TPS 데이터 수집
        turn_details = []  # 턴별 상세 데이터 수집 (추가됨)

        # 통계용 데이터 (warmup/cooldown 제외)
        stable_durations = []
        stable_ttft = []
        stable_tps = []

        # 제외된 턴 수
        excluded_by_min_tokens = 0

        # 통계 범위 계산
        total_turns = len(turns_data)
        stats_start = warmup_turns  # N번째 턴부터 통계
        stats_end = total_turns - cooldown_turns  # 뒤에서 N번째 턴까지

        for i, turn_data in enumerate(turns_data, 1):
            # 조기 종료 여부 확인
            if stop_event and stop_event.is_set():
                logger.info(f"⏹️  [{session_index}] 중지 신호 수신, 완료 {i-1}/{len(turns_data)}턴")
                break

            # 실제 턴 번호 계산 (원본 세션 기준)
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

            # 안정 구간 통계 여부 (warmup/cooldown 제외)
            turn_index = i - 1  # 0-based 인덱스로 변환
            is_stable_phase = stats_start <= turn_index < stats_end

            # 턴별 상세 데이터 기록 (실제 턴 번호 사용)
            turn_detail = {
                'turn_number': actual_turn_number,
                'success': result['success'],
                'duration_ms': result['duration_ms'],
                'input_tokens': result['input_tokens'],
                'output_tokens': result['output_tokens'],
                'ttft_ms': result.get('ttft_ms'),
                'tps': result.get('tps'),
                'response_text': result.get('response_text', ''),  # 응답 내용 추가
                'is_stable_phase': is_stable_phase,
                'error': result.get('error')
            }
            turn_details.append(turn_detail)

            if result['success']:
                successful += 1
                total_input += result['input_tokens']
                total_output += result['output_tokens']

                # 최소 출력 토큰 조건 충족 여부
                output_tokens = result['output_tokens']
                meets_min_tokens = output_tokens >= min_output_tokens if min_output_tokens > 0 else True

                # 전체 데이터 수집
                if result.get('ttft_ms') is not None:
                    all_ttft.append(result['ttft_ms'])
                if result.get('tps') is not None:
                    all_tps.append(result['tps'])

                # 안정 구간이면서 최소 토큰 조건 충족 시에만 통계 데이터 수집
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

            # 속도 제한 지연
            if rate_limit_delay > 0 and i < len(turns_data):
                await asyncio.sleep(rate_limit_delay)

        end_time = datetime.now()
        total_duration = sum(all_durations)
        success_rate = (successful / len(turns_data) * 100) if turns_data else 0.0

        # 안정 구간 데이터로 통계 계산 (있으면), 없으면 전체 데이터 사용
        ttft_data = stable_ttft if stable_ttft else all_ttft
        tps_data = stable_tps if stable_tps else all_tps

        avg_ttft = sum(ttft_data) / len(ttft_data) if ttft_data else None
        median_ttft = sorted(ttft_data)[len(ttft_data) // 2] if ttft_data else None
        avg_tps = sum(tps_data) / len(tps_data) if tps_data else None
        median_tps = sorted(tps_data)[len(tps_data) // 2] if tps_data else None

        # 로그 출력
        if warmup_turns > 0 or cooldown_turns > 0 or min_output_tokens > 0:
            log_msg = f"✅ [{session_index}] 완료: 성공률 {success_rate:.1f}%, 소요 {total_duration:.0f}ms "
            log_details = []

            if warmup_turns > 0 or cooldown_turns > 0:
                log_details.append(f"통계 범위: 제{stats_start+1}-{stats_end}턴")

            if min_output_tokens > 0 and excluded_by_min_tokens > 0:
                log_details.append(f"{min_output_tokens}토큰 미만 턴 제외: {excluded_by_min_tokens}개")

            if log_details:
                log_msg += f"({', '.join(log_details)}, 유효 샘플 {len(stable_ttft)}개)"
            else:
                log_msg += f"(유효 샘플 {len(stable_ttft)}개)"

            logger.info(log_msg)
        else:
            logger.info(f"✅ [{session_index}] 완료: 성공률 {success_rate:.1f}%, 소요 {total_duration:.0f}ms")

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
            turn_details=turn_details,  # 턴별 상세 데이터 추가
            errors=errors[:10]  # 최대 10개 오류만 유지
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
        다중 세션 동시 테스트
        Args:
            sessions_data: 세션 데이터 목록
            max_concurrent: 최대 동시 세션 수
            max_turns_per_session: 세션당 최대 테스트 턴 수
            rate_limit_delay: 요청 간 지연(초), 속도 제한 회피용
            warmup_turns: 각 세션 앞 N턴 통계 제외 (웜업 단계)
            cooldown_turns: 각 세션 뒤 N턴 통계 제외 (쿨다운 단계)
            min_output_tokens: 출력 토큰이 이 값보다 적으면 통계 제외 (기본 0=전체 포함)
            skip_first_turns: 각 세션 앞 N턴 건너뛰기, 요청 안 함 (기본 0)
            min_concurrent: 남은 활성 세션 수가 이 값보다 적을 때 테스트 중단
        """
        test_start = time.perf_counter()

        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 동시 테스트 시작")
        logger.info(f"{'='*80}")
        logger.info(f"  총 세션 수: {len(sessions_data)}")
        logger.info(f"  최대 동시 수: {max_concurrent}")
        logger.info(f"  세션당 최대 테스트: {max_turns_per_session or '전체'} 턴")
        if skip_first_turns > 0:
            logger.info(f"  앞 N턴 건너뛰기: {skip_first_turns} (요청 안 함)")
        logger.info(f"  요청 지연: {rate_limit_delay}s")
        if warmup_turns > 0 or cooldown_turns > 0:
            logger.info(f"  통계 범위: 앞 {warmup_turns}턴, 뒤 {cooldown_turns}턴 제외")
        if min_output_tokens > 0:
            logger.info(f"  최소 출력 토큰: {min_output_tokens} (이보다 적으면 통계 제외)")
        if min_concurrent:
            logger.info(f"  최소 동시 수: {min_concurrent} (이보다 낮으면 테스트 중단)")
        logger.info(f"{'='*80}\n")

        # 중지 이벤트 생성 (모든 세션에 중지 알림용)
        stop_event = asyncio.Event()

        # 활성 세션 수 추적용 원자 카운터
        active_sessions = {'count': len(sessions_data)}
        active_lock = asyncio.Lock()

        # 동시 수 제어용 세마포어 생성
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
                    # 세션 완료, 활성 수 갱신
                    async with active_lock:
                        active_sessions['count'] -= 1
                        remaining = active_sessions['count']

                        # 중지 트리거 여부 확인
                        if min_concurrent and remaining < min_concurrent and remaining > 0:
                            if not stop_event.is_set():
                                logger.warning(f"\n⚠️  남은 세션 수 ({remaining})가 임계값 ({min_concurrent}) 미만, 중지 신호 발생")
                                stop_event.set()

        # 모든 태스크 생성
        tasks = [
            test_with_semaphore(session, i)
            for i, session in enumerate(sessions_data, 1)
        ]

        # 모든 태스크 실행
        session_metrics = await asyncio.gather(*tasks)

        test_duration = (time.perf_counter() - test_start) * 1000

        # 통계 집계
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
        """테스트 보고서 생성"""
        total_requests = sum(s.tested_turns for s in session_metrics)
        successful_requests = sum(s.successful_turns for s in session_metrics)
        failed_requests = sum(s.failed_turns for s in session_metrics)

        total_input_tokens = sum(s.total_input_tokens for s in session_metrics)
        total_output_tokens = sum(s.total_output_tokens for s in session_metrics)

        # QPS 계산
        qps = total_requests / (test_duration_ms / 1000) if test_duration_ms > 0 else 0

        # 응답 시간 계산 (간이, 세션 평균 시간 사용)
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

        # 오류 요약
        total_errors = sum(len(s.errors) for s in session_metrics)
        error_types = {}
        for s in session_metrics:
            for error in s.errors:
                error_type = error.split(':')[0] if ':' in error else 'Unknown'
                error_types[error_type] = error_types.get(error_type, 0) + 1

        # TTFT 및 TPS 지표 집계
        all_session_ttft = [s.avg_ttft_ms for s in session_metrics if s.avg_ttft_ms is not None]
        all_session_tps = [s.avg_tps for s in session_metrics if s.avg_tps is not None]

        # 전역 TTFT 통계 계산 (단순 평균, TTFT는 샘플 수 영향 없음)
        avg_ttft = sum(all_session_ttft) / len(all_session_ttft) if all_session_ttft else None
        median_ttft = sorted(all_session_ttft)[len(all_session_ttft) // 2] if all_session_ttft else None
        p95_ttft_idx = int(len(all_session_ttft) * 0.95)
        p95_ttft = sorted(all_session_ttft)[p95_ttft_idx] if all_session_ttft else None

        # 전역 TPS 통계 계산 (가중 평균 사용)
        # 세션별 유효 샘플 수를 가중치로 사용
        sessions_with_tps = [s for s in session_metrics if s.avg_tps is not None and s.valid_tps_samples > 0]

        if sessions_with_tps:
            # 가중 평균 TPS
            total_weighted_tps = sum(s.avg_tps * s.valid_tps_samples for s in sessions_with_tps)
            total_samples = sum(s.valid_tps_samples for s in sessions_with_tps)
            avg_tps = total_weighted_tps / total_samples if total_samples > 0 else None

            # 중앙값: 모든 세션 TPS 샘플을 펼쳐 계산 (근사, 세션 평균 TPS 사용)
            # 참고: 단순화 버전이며, 이상적으로는 모든 단일 요청 TPS를 수집해야 함
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
        """테스트 보고서 출력"""
        print("\n" + "="*80)
        print("📊 동시성 테스트 보고서")
        print("="*80)

        print(f"\n🎯 테스트 설정:")
        print(f"  제공자: {report.provider_name}")
        print(f"  모델: {report.model_name}")
        print(f"  API URL: {report.api_url}")
        print(f"  테스트 시간: {report.test_time}")

        print(f"\n⚙️  동시성 설정:")
        print(f"  총 세션 수: {report.total_sessions}")
        print(f"  최대 동시 수: {report.max_concurrent_sessions}")
        print(f"  세션당 턴 수: {report.max_turns_per_session or '전체'}")

        print(f"\n📈 전체 통계:")
        print(f"  총 테스트 시간: {report.total_test_duration_ms / 1000:.2f}s")
        print(f"  총 요청 수: {report.total_requests}")
        print(f"  성공 요청: {report.successful_requests}")
        print(f"  실패 요청: {report.failed_requests}")
        print(f"  성공률: {report.overall_success_rate:.1f}%")

        print(f"\n⚡ 성능 지표:")
        print(f"  QPS (요청/초): {report.requests_per_second:.2f}")
        print(f"  평균 응답 시간: {report.avg_response_time_ms:.0f}ms")
        print(f"  P50 응답 시간: {report.p50_response_time_ms:.0f}ms")
        print(f"  P95 응답 시간: {report.p95_response_time_ms:.0f}ms")
        print(f"  P99 응답 시간: {report.p99_response_time_ms:.0f}ms")

        # TTFT 및 TPS 지표 표시
        if report.avg_ttft_ms is not None or report.avg_tps is not None:
            print(f"\n🚀 TTFT 및 TPS 지표:")
            if report.avg_ttft_ms is not None:
                print(f"  평균 TTFT: {report.avg_ttft_ms:.0f}ms")
                if report.median_ttft_ms is not None:
                    print(f"  중앙 TTFT: {report.median_ttft_ms:.0f}ms")
                if report.p95_ttft_ms is not None:
                    print(f"  P95 TTFT: {report.p95_ttft_ms:.0f}ms")
            if report.avg_tps is not None:
                print(f"  평균 TPS: {report.avg_tps:.2f} tokens/s")
                if report.median_tps is not None:
                    print(f"  중앙 TPS: {report.median_tps:.2f} tokens/s")

        print(f"\n🎯 토큰 통계:")
        print(f"  입력 토큰: {report.total_input_tokens:,}")
        print(f"  출력 토큰: {report.total_output_tokens:,}")
        print(f"  총 토큰: {report.total_tokens:,}")

        if report.total_errors > 0:
            print(f"\n⚠️  오류 통계:")
            print(f"  총 오류 수: {report.total_errors}")
            print(f"  오류 유형:")
            for error_type, count in sorted(report.error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"    {error_type}: {count}")

        print("\n" + "="*80 + "\n")

    def save_report(self, report: ConcurrentTestReport, output_file: str):
        """테스트 보고서 저장"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)

        logger.info(f"💾 보고서 저장됨: {output_path}")

    def generate_tps_distribution_chart(
        self,
        report: ConcurrentTestReport,
        output_dir: str,
        chart_format: str = 'both',
        show_content_threshold: int = 100
    ):
        """
        TPS 분포 곡선 차트 생성
        Args:
            report: 테스트 보고서
            output_dir: 출력 디렉터리
            chart_format: 차트 형식 ('matplotlib', 'plotly', 'both')
            show_content_threshold: 출력 토큰이 이 값보다 적을 때 내용 표시 (0=미표시)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 세션별 TPS 곡선 생성
        if chart_format in ['matplotlib', 'both'] and MATPLOTLIB_AVAILABLE:
            self._generate_matplotlib_charts(report, output_path, show_content_threshold)

        if chart_format in ['plotly', 'both'] and PLOTLY_AVAILABLE:
            self._generate_plotly_charts(report, output_path, show_content_threshold)

    def _generate_matplotlib_charts(self, report: ConcurrentTestReport, output_path: Path, show_content_threshold: int = 100):
        """matplotlib로 TPS 분포 곡선 생성"""
        logger.info("📊 matplotlib 차트 생성 중...")

        # 모든 세션을 하나의 큰 차트로
        num_sessions = len(report.sessions)
        fig, axes = plt.subplots(
            num_sessions, 1,
            figsize=(14, 4 * num_sessions),
            squeeze=False
        )

        for idx, session in enumerate(report.sessions):
            ax = axes[idx, 0]

            # 전체 턴 데이터 추출
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
                # 안정 구간 데이터 그리기
                if turn_data['stable']:
                    stable_points = turn_data['stable']
                    ax.plot([p['turn'] for p in stable_points],
                           [p['tps'] for p in stable_points],
                           'o-', color='#2E86AB', linewidth=2, markersize=6,
                           label='Stable Phase', alpha=0.8)

                # 웜업/쿨다운 구간 데이터 그리기
                if turn_data['warmup']:
                    warmup_points = turn_data['warmup']
                    ax.plot([p['turn'] for p in warmup_points],
                           [p['tps'] for p in warmup_points],
                           'o--', color='#A23B72', linewidth=1.5, markersize=4,
                           label='Warmup/Cooldown', alpha=0.6)

                # 평균선 추가
                if session.avg_tps:
                    ax.axhline(y=session.avg_tps, color='#F18F01',
                              linestyle='--', linewidth=2,
                              label=f'Avg TPS: {session.avg_tps:.2f}', alpha=0.8)

                # 통계 정보 계산
                all_points = turn_data['stable'] + turn_data['warmup']
                avg_ttft = sum(p['ttft'] for p in all_points) / len(all_points)
                avg_output_tokens = sum(p['output_tokens'] for p in all_points) / len(all_points)

                # 제목 및 통계 정보 설정
                title = f'Session {session.session_index}: {session.title[:50]}\n'
                title += f'Avg TTFT: {avg_ttft:.1f}ms | Avg Output: {avg_output_tokens:.0f} tokens'
                ax.set_title(title, fontsize=10, fontweight='bold', pad=10)

                ax.set_xlabel('Turn Number', fontsize=10)
                ax.set_ylabel('TPS (tokens/s)', fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(loc='best', fontsize=9)

                # y축 범위 설정 (과도한 요동 방지)
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

        # 차트 저장
        chart_file = output_path / 'tps_distribution_matplotlib.png'
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  ✅ Matplotlib 차트 저장됨: {chart_file}")

    def _generate_plotly_charts(self, report: ConcurrentTestReport, output_path: Path, show_content_threshold: int = 100):
        """plotly로 TPS 분포 곡선 생성 (인터랙티브)"""
        logger.info("📊 Plotly 인터랙티브 차트 생성 중...")

        # 서브플롯 생성
        num_sessions = len(report.sessions)
        fig = make_subplots(
            rows=num_sessions, cols=1,
            subplot_titles=[f'Session {s.session_index}: {s.title[:50]}'
                          for s in report.sessions],
            vertical_spacing=0.08 / num_sessions if num_sessions > 1 else 0.1
        )

        for idx, session in enumerate(report.sessions, 1):
            # 전체 턴 데이터 추출
            turn_data = {'stable': [], 'warmup': []}

            for turn_detail in session.turn_details:
                if turn_detail['success'] and turn_detail['tps'] is not None:
                    # 응답 내용 잘라내기 및 이스케이프
                    response_text = turn_detail.get('response_text', '')
                    output_tokens = turn_detail.get('output_tokens', 0)

                    # 임계값 조건을 만족하고 내용이 있으면 잘라서 표시
                    display_text = ''
                    if show_content_threshold > 0 and output_tokens < show_content_threshold and response_text:
                        # 앞뒤 공백 제거
                        response_text = response_text.strip()
                        if response_text:  # 빈 문자열이 아님 확인
                            # 최대 300자로 자르기 (hover 상자 과대 방지)
                            display_text = response_text[:300]
                            if len(response_text) > 300:
                                display_text += '...'
                            # HTML 특수문자 이스케이프
                            display_text = display_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                            display_text = display_text.replace('\n', '<br>')  # 줄바꿈을 HTML로
                            # 구분선 및 제목 추가
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
                # 안정 구간 곡선 추가
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
                                '%{customdata[3]}'  # 응답 내용 (있는 경우)
                                '<extra></extra>'
                            )
                        ),
                        row=idx, col=1
                    )

                # 웜업/쿨다운 구간 곡선 추가
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
                                '%{customdata[3]}'  # 응답 내용 (있는 경우)
                                '<extra></extra>'
                            )
                        ),
                        row=idx, col=1
                    )

                # 평균선 추가
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

            # 축 레이블 업데이트
            fig.update_xaxes(title_text='Turn Number', row=idx, col=1)
            fig.update_yaxes(title_text='TPS (tokens/s)', row=idx, col=1)

        # 레이아웃 업데이트
        fig.update_layout(
            title_text=f'TPS Distribution - {report.provider_name} ({report.model_name})',
            height=400 * num_sessions,
            showlegend=True,
            hovermode='closest',  # 긴 내용 표시에 적합한 closest 모드 사용
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="monospace",
                align="left",
                namelength=-1  # label 잘라내지 않음
            )
        )

        # HTML 파일 저장, hover 상자 크기 제어용 CSS 추가
        html_file = output_path / 'tps_distribution_plotly.html'

        # HTML 생성 및 사용자 CSS 추가
        html_string = fig.to_html(include_plotlyjs='cdn')

        # hover 상자 크기 제한 및 스크롤용 사용자 CSS 삽입
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
        # </head> 태그 앞에 CSS 삽입
        html_string = html_string.replace('</head>', custom_css + '</head>')

        # 파일에 쓰기
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_string)

        logger.info(f"  ✅ Plotly 차트 저장됨: {html_file}")


async def main():
    parser = argparse.ArgumentParser(
        description="동시 다중 세션 재생 테스트 도구"
    )
    parser.add_argument(
        '--input',
        required=True,
        help='입력용 다중 세션 JSON 파일'
    )
    parser.add_argument(
        '--num-sessions',
        type=int,
        help='테스트할 세션 수 (기본: 전체)'
    )
    parser.add_argument(
        '--selection-mode',
        choices=['first', 'random'],
        default='first',
        help='세션 선택 모드: first=처음 N개, random=랜덤 N개 (기본 first)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        help='랜덤 선택 시드 (재현 가능한 선택용)'
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
        help='모델 이름'
    )
    parser.add_argument(
        '--provider',
        default='Test Provider',
        help='공급자 이름'
    )
    parser.add_argument(
        '--api-format',
        choices=['anthropic', 'openai'],
        default='anthropic',
        help='API 형식'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=3,
        help='최대 동시 세션 수 (기본 3)'
    )
    parser.add_argument(
        '--max-turns',
        type=int,
        help='세션당 최대 테스트 턴 수'
    )
    parser.add_argument(
        '--rate-limit-delay',
        type=float,
        default=0.5,
        help='요청 간 지연 (초, 기본 0.5)'
    )
    parser.add_argument(
        '--warmup-turns',
        type=int,
        default=0,
        help='각 세션 앞 N턴 통계 제외 (웜업, 기본 0)'
    )
    parser.add_argument(
        '--cooldown-turns',
        type=int,
        default=0,
        help='각 세션 뒤 N턴 통계 제외 (쿨다운, 기본 0)'
    )
    parser.add_argument(
        '--min-concurrent',
        type=int,
        help='남은 활성 세션 수가 이보다 적을 때 테스트 중단 (선택, 저동시로 인한 TPS 이상 방지)'
    )
    parser.add_argument(
        '--min-output-tokens',
        type=int,
        default=16,
        help='출력 토큰이 이보다 적으면 통계 제외 (기본 16, 0=전체 포함)'
    )
    parser.add_argument(
        '--skip-first-turns',
        type=int,
        default=0,
        help='각 세션 앞 N턴 건너뛰기, 요청 안 함 (기본 0)'
    )
    parser.add_argument(
        '--output',
        help='출력 보고서 파일 경로'
    )
    parser.add_argument(
        '--generate-charts',
        action='store_true',
        help='TPS 분포 곡선 차트 생성'
    )
    parser.add_argument(
        '--chart-format',
        choices=['matplotlib', 'plotly', 'both'],
        default='both',
        help='차트 형식 (기본 both)'
    )
    parser.add_argument(
        '--show-content-threshold',
        type=int,
        default=100,
        help='출력 토큰이 이 임계값보다 작을 때 차트에 응답 내용 표시 (기본 100, 0=미표시)'
    )
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        help='반복 패널티 (vLLM 등, 보통 > 1.0)'
    )
    parser.add_argument(
        '--frequency-penalty',
        type=float,
        help='빈도 패널티 (OpenAI 표준, -2.0 ~ 2.0)'
    )
    parser.add_argument(
        '--presence-penalty',
        type=float,
        help='존재 패널티 (OpenAI 표준, -2.0 ~ 2.0)'
    )

    args = parser.parse_args()

    # 테스터 생성
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

    # 세션 데이터 로드
    data = tester.load_sessions_data(args.input)
    sessions = data.get('sessions', [])

    if not sessions:
        logger.error("❌ 세션 데이터를 찾을 수 없습니다")
        return 1

    # 테스트할 세션 선택
    sessions = tester.select_sessions(
        sessions=sessions,
        num_sessions=args.num_sessions,
        selection_mode=args.selection_mode,
        random_seed=args.random_seed
    )

    # 동시 테스트 실행
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

    # 보고서 출력
    tester.print_report(report)

    # 출력 디렉터리 결정
    if args.output:
        output_file = args.output
        output_dir = str(Path(output_file).parent)
    else:
        # 기본 출력 파일명
        provider = args.provider.replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = "benchmark_results"
        output_file = f"{output_dir}/concurrent_test_{provider}_{timestamp}.json"

    # 보고서 저장
    tester.save_report(report, output_file)

    # TPS 분포 곡선 차트 생성
    if args.generate_charts:
        logger.info("\n" + "="*80)
        logger.info("📊 TPS 분포 곡선 차트 생성")
        if args.show_content_threshold > 0:
            logger.info(f"  📝 출력 토큰 수 < {args.show_content_threshold} 인 응답 내용 표시")
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