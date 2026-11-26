from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

class KoreanSingleExtractionOutput(BaseModel):
    """국어 기출 단일 문항 추출 스키마 (독서·문학 공통 영역 지문 세트 중심 추출)."""

    korean_section: Literal["독서", "문학", "화법과 작문", "언어와 매체"] = Field(
        ...,
        description="영역 구분 ('독서' | '문학' | '화법과 작문' | '언어와 매체')",
    )
    korean_subgenre: str = Field(
        default="",
        description="세부 갈래 (예: 독서론, 인문·예술, 사회·경제, 과학·기술, 주제 통합, 현대시, 고전시가, 현대소설, 고전소설, 극·수필, 갈래 복합)",
    )
    question_type: str = Field(
        default="",
        description="원래 문항 유형 (예: 세부 내용 일치, 핵심 논지 추론, <보기> 사례 적용, 서술상 특징, 표현상 특징, 문맥적 어휘 의미)",
    )
    passage_group_id: str = Field(
        default="",
        description="공통 지문 세트 범위 (예: '1~3', '4~9', '18~21', 단독 문항은 해당 번호 문자열)",
    )
    passage_text: str = Field(
        default="",
        description="발문과 선지를 제외한 순수 국어 지문 원문 ((가)/(나) 구분, [A]~[E], ㉠~㉤, ⓐ~ⓔ 표지 포함; 화작/언매는 빈 문자열)",
    )
    passage_topic: str = Field(
        default="",
        description="지문 핵심 주제 또는 작품명·작가 1줄 요약 (예: '박완서, 「나목」', '카르납의 논리실증주의')",
    )
    key_vocabulary: list[str] = Field(
        default_factory=list,
        description="지문 내 핵심 어휘·문학적 개념어 3~6개 목록",
    )
    official_explanation: str = Field(
        ...,
        description="해설지 기준 핵심 풀이 및 정답 근거 요약",
    )
    answer: str = Field(
        ...,
        description="공식 정답 ('1'~'5' 단일 숫자 문자열)",
    )
    standard_code: str = Field(
        default="",
        description="2022 개정 교육과정 성취기준 대표 코드 (기준표가 제공된 경우에만 매칭, 없으면 빈 문자열)",
    )
    standard_codes: list[str] = Field(
        default_factory=list,
        description="2022 개정 교육과정 관련 성취기준 코드 목록",
    )
    needs_review: bool = Field(
        default=False,
        description="성취기준 코드 검증 필요 여부",
    )
    review_notes: list[str] = Field(
        default_factory=list,
        description="검증 시 발견된 주의사항",
    )


class KoreanQuestionExtractionItem(KoreanSingleExtractionOutput):
    """국어 모의고사 전체 PDF 일괄 추출 시 문항 번호가 포함된 항목."""

    q_num: int = Field(..., ge=1, le=45, description="문항 번호 (1~45)")


class BatchKoreanExamExtractionOutput(BaseModel):
    """국어 모의고사 PDF 전체 일괄 추출 출력 스키마."""

    questions: list[KoreanQuestionExtractionItem] = Field(default_factory=list)


class KoreanSetQuestionOutput(BaseModel):
    """국어 지문 세트 전체 재구성 시 생성되는 개별 문항."""

    slot_index: int = Field(..., ge=1, description="세트 내 문항 순번 (1부터 시작)")
    question_type_label: str = Field(
        ...,
        description="새로 출제한 문항 유형 (예: 세부 내용 일치, 핵심 논지 추론, <보기> 적용, 표현·서술상 특징, 문맥 어휘)",
    )
    stem: str = Field(
        ...,
        description="순수 발문 1문장만 작성 (<보기> 본문 및 지문 본문 미포함, 불필요한 기호 괄호 풀이 금지)",
    )
    bogi_text: str = Field(
        default="",
        description="<보기> 박스 내부 본문 (필요한 문항만 작성, 일반 문항은 빈 문자열)",
    )
    choices: list[str] = Field(
        ...,
        description="①~⑤ 선지 번호를 제외한 5개 선지 텍스트 리스트 (마크다운 ** 금지, 어휘 문항 목표 어휘만 <u>...</u> 밑줄 허용)",
    )
    answer: str = Field(..., description="정답 ('1'~'5' 단일 숫자 문자열)")
    explanation: str = Field(
        default="",
        description="정답 핵심 근거 요약 (평어체 '~다.')",
    )
    choice_explanations: list[str] = Field(
        default_factory=list,
        description="choices[0]~choices[4] 순서와 1:1 대응하는 5개 선지별 정오 근거 리스트 (평어체 '~다.')",
    )


class KoreanSetReconstructionOutput(BaseModel):
    """국어 지문 1개 세트에 대한 전체 문항(3~6문항) 자유 유형 재구성 출력 스키마."""

    set_intro: str = Field(
        ...,
        description="세트 공통 발문 (예: '다음 글을 읽고 물음에 답하시오.')",
    )
    questions: list[KoreanSetQuestionOutput] = Field(default_factory=list)


class KoreanBlindQuestionVerdict(BaseModel):
    """정답 은닉 Blind Solver의 개별 국어 문항 검수 판정."""

    slot_index: int = Field(..., ge=1, description="세트 내 문항 순번")
    selected_answer: Literal["1", "2", "3", "4", "5"] = Field(
        ...,
        description="정답을 보지 않고 지문·발문·선지만 읽고 독립적으로 도출한 유일한 정답 ('1'~'5')",
    )
    ambiguous_choices: list[str] = Field(
        default_factory=list,
        description="복수 정답 소지가 있거나 경합하는 선지 번호 목록 (예: ['2', '4'])",
    )
    has_flaw: bool = Field(
        default=False,
        description="지문 사실 불일치, 지문 미존재 표지 참조, 주술 호응/비문, 자체모순 선지, <보기> 오류 등 출제 결함 여부",
    )
    flaw_reason: str = Field(
        default="",
        description="결함 또는 정답 경합 사유 (문제 없으면 빈 문자열)",
    )


class KoreanBlindSetVerdict(BaseModel):
    """국어 세트 전체에 대한 Blind Solver 일괄 검수 결과."""

    verdicts: list[KoreanBlindQuestionVerdict] = Field(default_factory=list)


class KoreanSetRepairOutput(BaseModel):
    """결함이 발견된 국어 세트 문항 슬롯만 정밀 재출제하는 출력 스키마."""

    questions: list[KoreanSetQuestionOutput] = Field(default_factory=list)


class KoreanParagraphSummary(BaseModel):
    """국어 지문의 문단/연/장면별 핵심 요약."""

    label: str = Field(..., description="구분 라벨 (예: '1문단', '(가) 1연', '[A] 부분')")
    summary: str = Field(..., description="해당 문단/연의 핵심 내용 1~2문장 요약")
    key_point: str = Field(default="", description="핵심 논지·전개 기능 또는 표현 포인트 1줄")


class KoreanMarkerVocabItem(BaseModel):
    """지문 내 기호(㉠~㉤, ⓐ~ⓔ) 및 핵심 어휘/개념 해설."""

    term: str = Field(..., description="기호 또는 핵심 어휘 (예: '㉠', 'ⓐ 반증', '의제와 간주')")
    meaning: str = Field(..., description="문맥적 의미, 지시 대상 또는 개념 설명")


class KoreanPoetryLineNote(BaseModel):
    """시(현대시·고전시가 등) 원문에서 밑줄을 그어 해설할 핵심 시구·시어 항목."""

    quote: str = Field(..., description="원문에 실제로 등장하는 시구·시어 그대로 발췌 (1개 행 또는 구절)")
    explanation: str = Field(..., description="해당 밑줄 시구의 표현상 특징, 시적 의미 및 화자의 정서·태도 해설")


class KoreanPoetryMark(BaseModel):
    """시 원문 특정 행·시어에 직접 그리는 밑줄·물결선·네모·동그라미·세모 및 행간 주석."""

    line_index: int = Field(..., description="해당 시 작품 `lines` 리스트 내 1부터 시작하는 행 번호 (빈 줄 포함)")
    quote: str = Field(..., description="해당 행 원문에 실제로 존재하는 시어·시구 그대로 발췌")
    mark_type: Literal["underline", "wavy", "box", "circle", "triangle"] = Field(
        default="underline",
        description="표기 도형: underline(직선 밑줄), wavy(물결선), box(네모 박스), circle(동그라미), triangle(세모/시련·부정)",
    )
    color: Literal["red", "blue", "dark"] = Field(
        default="red",
        description="주석 색상: red(화자 정서·감각 이미지·부정/시련), blue(구조적 특징·긍정/희망/극복 의지), dark(기본 대상/지시어)",
    )
    position: Literal["above", "below"] = Field(
        default="below",
        description="행간 주석 배치 위치 (below=행 아래, above=행 위)",
    )
    note: str = Field(..., description="시어 위/아래 행간에 화살표로 연결할 짧은 핵심 해설 (25자 이내 권장)")


class KoreanPoetryMarginBracket(BaseModel):
    """시 좌·우 여백에 여러 행/연을 묶어 표시하는 구조 브라켓·수미상관·시상전환 주석."""

    start_line: int = Field(..., description="묶음 시작 행 번호 (1부터 시작)")
    end_line: int = Field(..., description="묶음 끝 행 번호 (1부터 시작)")
    side: Literal["left", "right"] = Field(
        default="right",
        description="여백 위치 (right=연/행 묶음·대구·시상전환 등, left=수미상관 큰 연결선 등)",
    )
    style: Literal["brace", "arc", "divider"] = Field(
        default="brace",
        description="선 스타일: brace(중괄호 } / { 묶음), arc(두 연을 잇는 큰 호 - 수미상관 등), divider(시상 전환 점선 구분선)",
    )
    color: Literal["red", "blue", "dark"] = Field(
        default="blue",
        description="여백 주석 색상 (blue/red/dark)",
    )
    note: str = Field(..., description="여백에 표시할 구조·시상 전개 핵심 주석 (예: '대구법 & 연쇄법', '수미상관 구조', '시상 전환')")


class KoreanPoetryWorkAnnotation(BaseModel):
    """(가)/(나) 등 개별 시 작품 1편의 행간·여백 주석형 원문 분석 데이터."""

    work_label: str = Field(default="", description="작품 구분 기호 (예: '(가)', '(나)', 단일 작품이면 '')")
    title_author: str = Field(..., description="작품명 및 작가 (예: '신동엽, 「산에 언덕에」')")
    lines: list[str] = Field(
        default_factory=list,
        description="원문 시의 각 행 리스트 (연과 연 사이는 빈 문자열 '' 포함). 반드시 원문 표기 그대로 수록.",
    )
    marks: list[KoreanPoetryMark] = Field(
        default_factory=list,
        description="시어·시구별 밑줄/물결/네모/동그라미/세모 및 행간 주석 목록 (8~16개 권장)",
    )
    brackets: list[KoreanPoetryMarginBracket] = Field(
        default_factory=list,
        description="좌·우 여백 묶음 괄호, 수미상관 연결선, 시상 전환 구분선 목록 (2~5개 권장)",
    )


class KoreanDiagramNode(BaseModel):
    """비문학 지문 이해를 돕는 도식 그림의 개별 개념/단계 노드."""

    title: str = Field(..., description="카드 제목 (예: '1단계: 압력 상승', '현명주의')")
    subtitle: str = Field(default="", description="부제 또는 역할 태그 (예: '원인', '핵심 입장')")
    details: list[str] = Field(
        default_factory=list,
        description="카드 안에 들어갈 1~3줄 핵심 설명 (각 줄 28자 이내 권장)",
    )
    accent_color: Literal["blue", "green", "amber", "purple", "slate"] = Field(
        default="blue",
        description="카드 강조 색상 (blue / green / amber / purple / slate)",
    )


class KoreanMatrixRow(BaseModel):
    """비교 매트릭스표(`matrix`)의 1개 비교 기준 행."""

    criterion: str = Field(..., description="비교 기준 항목명 (예: '통치 목적', '법의 적용', '핵심 수단')")
    values: list[str] = Field(
        default_factory=list,
        description="상단 비교 대상(nodes 순서)별 대응 내용 목록 (각 항목 28자 이내 권장)",
    )


class KoreanGraphCurve(BaseModel):
    """X-Y 함수/비례 그래프(`graph`)의 개별 곡선·직선 정보."""

    label: str = Field(..., description="곡선/변수 이름 (예: '수요의 가격탄력성', '평균비용')")
    trend: Literal["up", "down", "u_shape", "inv_u", "flat"] = Field(
        default="up",
        description="그래프 추세 (up=우상향 비례, down=우하향 반비례, u_shape=U자형, inv_u=역U자형, flat=일정)",
    )
    accent_color: Literal["blue", "green", "amber", "purple", "slate"] = Field(default="blue")
    annotation: str = Field(default="", description="곡선 특징 설명 (예: 'X 증가 시 Y 감소')")


class KoreanGraphSpec(BaseModel):
    """X-Y 좌표 그래프(`graph`) 명세."""

    x_label: str = Field(..., description="가로축(X축) 변수명 (예: '수요의 가격탄력성')")
    y_label: str = Field(..., description="세로축(Y축) 변수명 (예: '이용료 수준')")
    curves: list[KoreanGraphCurve] = Field(
        default_factory=list,
        description="좌표평면에 그릴 1~2개 추세선",
    )
    highlight_point: str = Field(
        default="",
        description="균형점·임계점·교차점 설명 (예: '임계점: 간접 네트워크 외부성 극대화')",
    )


class KoreanCharacterEdge(BaseModel):
    """소설 인물관계도(`character_map`)의 인물 간 관계선 정보."""

    source_idx: int = Field(default=0, ge=0, description="출발 인물 노드 인덱스 (0-based)")
    target_idx: int = Field(default=1, ge=0, description="도착 인물 노드 인덱스 (0-based)")
    relation: str = Field(default="", description="관계 설명 (예: '조력', '모함·대립', '부자 갈등')")
    polarity: Literal["positive", "negative", "neutral"] = Field(
        default="neutral",
        description="positive=조력/지지(+), negative=갈등/대립(-), neutral=일반 관계",
    )


class KoreanEmotionPoint(BaseModel):
    """시상 전개 정서 변화 그래프(`emotion_arc`)의 구간별 정서 좌표."""

    stage: str = Field(..., description="구간·시점 라벨 (예: '1연', '전반부', '과거')")
    emotion: str = Field(..., description="해당 구간 핵심 정서·태도 (예: '절망·상실', '극복 의지')")
    level: int = Field(
        default=0,
        ge=-2,
        le=2,
        description="정서 극성 수준 (-2=극심한 부정, -1=부정, 0=중립/전환, 1=긍정, 2=강한 긍정/극복)",
    )


class KoreanVisualDiagram(BaseModel):
    """비문학·문학 지문의 핵심 원리·관계·정서·구조를 시각화하는 16종 특화 도식 그림 명세."""

    diagram_type: Literal[
        "flow",
        "matrix",
        "graph",
        "anatomy",
        "cycle",
        "timeline",
        "venn",
        "scale",
        "fishbone",
        "quadrant",
        "decision_tree",
        "pyramid",
        "character_map",
        "emotion_arc",
        "poetic_contrast",
        "space_journey",
        "comparison",
        "hierarchy",
    ] = Field(
        default="flow",
        description=(
            "도식 유형: 비문학 12종(flow, matrix, graph, anatomy, cycle, timeline, venn, scale, "
            "fishbone, quadrant, decision_tree, pyramid) + 문학 4종(character_map=인물관계·갈등도, "
            "emotion_arc=정서변화 그래프, poetic_contrast=(+)/(-) 시어 대립도, space_journey=공간·시선 이동 경로도)"
        ),
    )
    title: str = Field(..., description="도식 상단 제목 (예: '양면시장 가격탄력성과 이용료 반비례 관계')")
    center_label: str = Field(
        default="",
        description="anatomy(중앙 장치명), cycle(순환 주제), venn(교집합), fishbone(최종 결과/현상), decision_tree(핵심 조건 질문) 라벨",
    )
    quadrant_x_label: str = Field(
        default="",
        description="diagram_type='quadrant'일 때 가로축(X축) 분류 기준 (예: '배제성 (좌: 불가 ↔ 우: 가능)')",
    )
    quadrant_y_label: str = Field(
        default="",
        description="diagram_type='quadrant'일 때 세로축(Y축) 분류 기준 (예: '경합성 (하: 낮음 ↔ 상: 높음)')",
    )
    nodes: list[KoreanDiagramNode] = Field(
        default_factory=list,
        description="도식을 구성하는 핵심 항목/대상/부위/단계 카드 2~4개",
    )
    relation_labels: list[str] = Field(
        default_factory=list,
        description="노드 사이 화살표/연결선에 표시할 짧은 관계 라벨 목록 (각 8자 이내)",
    )
    matrix_rows: list[KoreanMatrixRow] = Field(
        default_factory=list,
        description="diagram_type='matrix'일 때 사용하는 기준별 비교 행 리스트 (3~5행)",
    )
    graph_spec: KoreanGraphSpec | None = Field(
        default=None,
        description="diagram_type='graph'일 때 사용하는 X-Y 좌표 그래프 명세",
    )
    scale_tilt: Literal["balanced", "left", "right"] = Field(
        default="balanced",
        description="diagram_type='scale'일 때 저울 기울기 (balanced=균형/절충, left=좌측 우위, right=우측 우위)",
    )
    character_edges: list[KoreanCharacterEdge] = Field(
        default_factory=list,
        description="diagram_type='character_map'일 때 인물 간 조력/갈등 관계선 목록",
    )
    emotion_points: list[KoreanEmotionPoint] = Field(
        default_factory=list,
        description="diagram_type='emotion_arc'일 때 연·시간별 정서 변화 좌표 목록 (3~5개)",
    )
    takeaway: str = Field(
        default="",
        description="도식 하단에 표시할 1줄 핵심 정리 (시험 출제 포인트)",
    )


class KoreanPassageAnalysisOutput(BaseModel):
    """수능·학원형 국어 지문 정밀 분석지 구조화 출력 스키마."""

    title: str = Field(..., description="분석지 제목 (예: '법적 의제와 간주의 차이' 또는 '오세영, 「보석 1」 외')")
    core_theme: str = Field(..., description="지문 전체의 핵심 주제 · 중심 화제(또는 작품 주제 의식) 1~2문장")
    structure_flow: str = Field(
        ...,
        description="지문 전체 논리 전개 구조 또는 작품 시상 전개·서사 흐름 요약 (화살표 → 활용)",
    )
    paragraph_summaries: list[KoreanParagraphSummary] = Field(
        default_factory=list,
        description="문단별(독서) 또는 연·장면별(문학) 핵심 요약 리스트",
    )
    marker_and_vocab_notes: list[KoreanMarkerVocabItem] = Field(
        default_factory=list,
        description="지문 속에서 '~~는 ~~~~이다' 형태로 정의가 제시된 핵심 개념어 및 정의 정리 리스트 (기호 표지 제외)",
    )
    area_specific_notes: list[str] = Field(
        default_factory=list,
        description="영역별 핵심 정리 (독서: 개념 대조·인과관계 / 문학: 화자·정서·태도·표현상 특징 / 화작·언매: 말하기 전략·문법 원리)",
    )
    exam_checkpoints: list[str] = Field(
        default_factory=list,
        description="수능·내신 출제 예상 포인트 및 오답 함정 주의사항 3~5개",
    )
    poetry_underline_notes: list[KoreanPoetryLineNote] = Field(
        default_factory=list,
        description="시(현대시·고전시가 등) 갈래일 때 원문에서 밑줄을 그어 해설할 핵심 시구·시어 목록 (4~7개). 산문/비문학은 빈 리스트.",
    )
    poetry_works: list[KoreanPoetryWorkAnnotation] = Field(
        default_factory=list,
        description="시(현대시·고전시가 등) 갈래일 때 (가)/(나) 작품별로 분리한 행간·여백 주석형 원문 분석 데이터 리스트. 산문/비문학은 빈 리스트.",
    )
    visual_diagram: KoreanVisualDiagram | None = Field(
        default=None,
        description="비문학(독서) 등 지문 이해를 돕는 개념·원리·비교 도식 그림 데이터 (불필요한 경우 None)",
    )
