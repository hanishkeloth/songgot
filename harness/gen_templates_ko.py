"""Template-generated Korean tool-calling data. No teacher model: every example is a Korean
utterance built from hand-written frames per tool and slot fillers (names, places, dates,
times, amounts), paired with the exact call. Diversity comes from 6 to 10 frames per tool,
register variants (반말/존댓말/격식/사투리/오타), slot fillers, and date words that resolve
against a fixed "today" (2026-09-10, Thursday).

    .venv/bin/python harness/gen_templates_ko.py --n 60000 --out data/ko_raw.jsonl

Output rows: {"tool": name, "query": text, "call": {"name", "arguments"}} (same as teacher_gen).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = {t["name"]: t for t in json.load(open(ROOT / "data" / "tools_ko.json", encoding="utf-8"))}
TODAY = dt.date(2026, 9, 10)

NAMES = ["민수", "지은", "서연", "준호", "하늘", "엄마", "아빠", "김 과장", "박 대리", "이 부장", "최 팀장", "동생", "할머니", "형", "누나", "언니", "오빠", "수진", "태양", "은우", "도윤", "나영"]
PLACES = ["강남역", "홍대입구역", "서울역", "인천공항", "부산역", "해운대", "광안리", "제주공항", "성수동", "판교", "여의도", "잠실", "수원역", "대전역", "광주송정역", "동대구역", "경복궁", "남산타워", "이태원", "건대입구", "회사", "집", "학교", "우리 동네 마트", "코엑스"]
CITIES = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "제주", "수원", "창원", "강릉", "춘천", "전주", "포항", "여수"]
WORLD = ["도쿄", "뉴욕", "런던", "파리", "시드니", "싱가포르", "방콕", "베를린", "로스앤젤레스", "두바이", "하노이", "타이베이"]
ROOMS = ["거실", "안방", "작은방", "주방", "서재", "아이 방", "현관", "화장실"]
SONGS = ["잔잔한 재즈", "신나는 케이팝", "공부할 때 듣는 노래", "드라이브 음악", "클래식", "힙합", "발라드", "빗소리", "카페 음악", "운동할 때 듣는 노래", "락", "트로트"]
FOODS = ["치킨", "피자", "떡볶이", "김밥", "짜장면", "족발", "초밥", "삼겹살", "냉면", "돈까스", "마라탕", "샐러드", "햄버거", "순대국", "파스타"]
PRODUCTS = ["무선 이어폰", "러닝화", "공기청정기", "캠핑 의자", "노트북 거치대", "전기포트", "요가 매트", "가습기", "선풍기", "책상 램프", "보조배터리", "우산"]
APPS = ["카메라", "계산기", "지도", "메모", "캘린더", "은행 앱", "음악", "유튜브", "설정", "갤러리", "날씨", "메일"]
MEDICINES = ["혈압약", "비타민", "감기약", "진통제", "유산균", "알레르기약", "철분제"]
DOCS = ["보고서 초안", "회의 자료", "견적서", "출장 정산", "주간 보고", "계약서 검토 요청"]
TASKS = ["보고서 마무리", "세탁물 찾기", "엄마한테 전화", "은행 가기", "우유 사기", "약 챙기기", "운동", "책 반납", "발표 준비", "택배 보내기"]
NOTE_BITS = ["우유 2개, 계란 한 판", "다음 주 발표 아이디어: 사용자 인터뷰 먼저", "주차 위치 지하 3층 B12", "와이파이 비밀번호는 바꿔야 함", "회의에서 나온 숙제 정리", "여행 준비물: 여권, 충전기, 우산"]
TITLES = ["팀 회의", "치과 예약", "친구 생일", "발표", "부모님 방문", "헬스", "면접", "스터디", "저녁 약속", "온라인 세미나"]
DRAMAS = ["새로 나온 드라마", "어제 못 본 예능", "다큐멘터리", "애니메이션", "코미디 영화", "지난주 축구 경기"]
WORDS = ["가치", "사과", "의의", "정성", "batch", "resilience", "감사", "허심탄회", "기특하다", "serendipity"]
EXPR = ["23 곱하기 17", "1200 나누기 8", "35만 원의 12퍼센트", "3.5 더하기 2.25", "2의 10제곱", "98765 빼기 4321"]
CURR = ["KRW", "USD", "JPY", "EUR", "CNY", "GBP"]
CURR_KO = {"KRW": "원", "USD": "달러", "JPY": "엔", "EUR": "유로", "CNY": "위안", "GBP": "파운드"}
UNIT_PAIRS = [("평", "제곱미터"), ("제곱미터", "평"), ("근", "그램"), ("섭씨", "화씨"), ("마일", "킬로미터"), ("인치", "센티미터"), ("파운드", "킬로그램")]
STOCKS = ["삼성전자", "현대차", "카카오", "네이버", "테슬라", "애플", "엔비디아", "LG에너지솔루션"]
ISSUE_KO = {"illegal_parking": "불법 주차", "road_damage": "도로 파손", "noise": "층간 소음", "garbage": "쓰레기 무단 투기", "streetlight": "가로등 고장"}
CERT_KO = {"resident_registration": "주민등록등본", "family_relation": "가족관계증명서", "income_certificate": "소득증명원", "business_registration": "사업자등록증"}
SVC_KO = {"community_center": "주민센터", "health_center": "보건소", "post_office": "우체국", "police": "경찰서", "fire_station": "소방서", "library": "도서관"}
CAT_KO = {"restaurant": "식당", "cafe": "카페", "pharmacy": "약국", "hospital": "병원", "convenience_store": "편의점", "parking": "주차장", "gas_station": "주유소", "ev_charger": "전기차 충전소", "bank": "은행", "subway_station": "지하철역", "bus_stop": "버스 정류장"}
APPL_KO = {"washer": "세탁기", "dryer": "건조기", "robot_vacuum": "로봇청소기", "dishwasher": "식기세척기", "air_purifier": "공기청정기"}
SET_KO = {"wifi": "와이파이", "bluetooth": "블루투스", "do_not_disturb": "방해금지 모드", "brightness": "화면 밝기", "power_saving": "절전 모드", "airplane_mode": "비행기 모드", "flashlight": "손전등"}
LANG_KO = {"en": "영어", "ja": "일본어", "zh": "중국어", "es": "스페인어", "fr": "프랑스어", "vi": "베트남어", "ko": "한국어"}
LEAVE_KO = {"annual": "연차", "half_day_am": "오전 반차", "half_day_pm": "오후 반차", "sick": "병가"}
METRIC_KO = {"weight_kg": ("체중", ["72.4", "58", "80.1", "63.5"]), "blood_pressure": ("혈압", ["120/80", "135/85", "118/76"]), "steps": ("걸음 수", ["8500", "12000", "6200"]), "sleep_hours": ("수면 시간", ["6.5", "7", "5"]), "water_ml": ("물 섭취량", ["500", "1500", "2000"])}

ENDINGS = {  # suffixes glued to a verb stem that already ends in 해/켜/꺼/알려/찾아/보내 etc.
    "banmal": ["줘", " 줘", "라", "줄래?", " 주라", "줘요"],
    "jondae": [" 주세요", "주세요", " 주실래요?", "주시겠어요?", " 주세요.", "주세요~"],
    "formal": [" 주십시오", " 주시기 바랍니다", " 주시겠습니까?", " 주십시오."],
}


def rel_date(rng):
    """Korean date word -> ISO date relative to TODAY."""
    opts = [("오늘", 0), ("내일", 1), ("모레", 2), ("글피", 3), ("다음 주 월요일", (7 - TODAY.weekday()) % 7 or 7), ("이번 주 금요일", (4 - TODAY.weekday()) % 7),
            ("이번 주 토요일", (5 - TODAY.weekday()) % 7), ("다음 주 수요일", ((2 - TODAY.weekday()) % 7) + 7), ("일주일 뒤", 7), ("보름 뒤", 15)]
    for d in range(11, 30):
        opts.append((f"{TODAY.month}월 {d}일", d - TODAY.day))
    for d in (1, 3, 5, 10, 15, 20, 25):
        opts.append((f"10월 {d}일", (dt.date(2026, 10, d) - TODAY).days))
    w, delta = rng.choice(opts)
    return w, (TODAY + dt.timedelta(days=delta)).isoformat()


def rel_time(rng):
    """Korean time phrase -> HH:MM."""
    h = rng.choice([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]); m = rng.choice([0, 0, 0, 30, 15, 45, 10, 20, 50])
    if h < 12:
        word = f"아침 {h}시" if h < 10 else f"오전 {h}시"
    elif h == 12:
        word = "낮 12시" if m == 0 else "12시"
    elif h < 18:
        word = f"오후 {h-12}시"
    else:
        word = f"저녁 {h-12}시" if h < 21 else f"밤 {h-12}시"
    if m == 30:
        word += " 반"
    elif m:
        word += f" {m}분"
    return word, f"{h:02d}:{m:02d}"


def money(rng):
    v = rng.choice([5000, 9900, 15000, 25000, 30000, 49000, 80000, 120000, 200000, 350000, 1000000])
    if v >= 10000 and v % 10000 == 0:
        w = f"{v//10000}만 원"
    elif v >= 10000:
        w = f"{v//10000}만 {v%10000}원"
    else:
        w = f"{v}원"
    return w, v


def num(rng, lo, hi):
    n = rng.randint(lo, hi); return str(n), n


def frames(name):
    """Korean utterance frames per tool. {x} slots are filled by fill(); {e} is the register ending."""
    F = {
        "send_chat_message": ["{who}한테 {msg}라고 {e}", "{who}에게 {msg} 메시지 보내{e}", "{msg}라고 {who}한테 카톡 보내{e}", "{who}한테 문자로 {msg} {e}", "{msg} 이렇게 {who}한테 전해{e}"],
        "make_call": ["{who}한테 전화 걸어{e}", "{who}에게 전화해{e}", "{who} 영상통화로 연결해{e}", "지금 {who}한테 전화 좀 걸어{e}", "{who} 번호로 통화 연결{e}"],
        "read_unread_messages": ["안 읽은 메시지 읽어{e}", "새로 온 카톡 있어? 읽어{e}", "읽지 않은 문자 {n}개만 보여{e}", "안 읽은 메일 읽어{e}", "새 메시지 확인해{e}"],
        "create_event": ["{date} {time}에 {title} 일정 추가해{e}", "{date} {time} {title} 캘린더에 넣어{e}", "{title}을 {date} {time}으로 잡아{e}", "{date} {time}에 {place}에서 {title} 일정 등록해{e}", "{date} {time} {who}랑 {title} 일정 만들어{e}"],
        "list_events": ["{date} 일정 뭐 있어?", "{date} 스케줄 알려{e}", "{date} 일정 보여{e}", "이번 주 일정 정리해{e}", "이번 달 일정 알려{e}"],
        "set_reminder": ["{date} {time}에 {task} 알림 설정해{e}", "{date} {time}에 {task} 알려{e}", "매일 {time}에 {task} 리마인드해{e}", "{task} {date} {time}으로 알림 맞춰{e}", "매주 {time}에 {task} 알림 걸어{e}"],
        "set_alarm": ["{time}에 알람 맞춰{e}", "{time} 알람 설정해{e}", "내일 {time}에 깨워{e}", "평일 {time}으로 알람 걸어{e}", "{time} {label} 알람 만들어{e}", "주말 {time}에 알람 켜{e}"],
        "set_timer": ["{n}분 타이머 맞춰{e}", "{n}분 뒤에 알려{e}", "{n}분 타이머 시작해{e}", "{label} {n}분 타이머 켜{e}", "{n}초 타이머 걸어{e}"],
        "get_world_time": ["{world} 지금 몇 시야?", "{world} 시간 알려{e}", "{world}은 지금 몇 시{e}", "현재 {world} 시각 확인해{e}"],
        "navigate_to": ["{place}까지 길 안내해{e}", "{place}로 가는 길 알려{e}", "{place} 내비 켜{e}", "{place}까지 {mode}로 안내해{e}", "{place}로 출발, 톨게이트는 피해{e}", "{place} 가자, 길 찾아{e}"],
        "find_nearby": ["근처 {cat} 찾아{e}", "가까운 {cat} 어디 있어?", "주변에 지금 문 연 {cat} 있어?", "{n}미터 안에 {cat} 검색해{e}", "이 근처 {cat} 알려{e}"],
        "estimate_travel_time": ["{place}까지 얼마나 걸려?", "{place}에서 {place2}까지 {mode}로 얼마나 걸려?", "{time}에 출발하면 {place}에 몇 시 도착이야?", "{place}까지 소요 시간 알려{e}", "{place2}에서 {place}까지 시간 계산해{e}"],
        "get_bus_arrival": ["{place} 정류장 버스 언제 와?", "{place} 정류장 {n}번 버스 도착 정보 알려{e}", "{place}에서 {n}번 버스 몇 분 남았어?", "{place} 정류장 도착 버스 확인해{e}"],
        "get_subway_arrival": ["{station}역 {line}호선 열차 언제 와?", "{station}역 지하철 도착 정보 알려{e}", "{station}역 상행 열차 몇 분 남았어?", "{station}역 하행 다음 열차 확인해{e}"],
        "book_train": ["{date} {city}에서 {city2} 가는 기차 예매해{e}", "{date} {time} {city}행 KTX {n}명 예매해{e}", "{city}에서 {city2}까지 {date} 기차표 끊어{e}", "{date} {city2}행 특실 {n}명 예약해{e}"],
        "call_taxi": ["{place}까지 택시 불러{e}", "택시 호출해{e}, 목적지는 {place}", "{place}로 가는 대형 택시 잡아{e}", "지금 여기서 {place}까지 택시 불러{e}", "{place}까지 모범택시 호출해{e}"],
        "get_weather": ["{city} 날씨 어때?", "{date} {city} 날씨 알려{e}", "{city} 오늘 비 와?", "{date} {city} 기온 어떻게 돼?", "{city} 주말 날씨 확인해{e}"],
        "get_air_quality": ["{city} 미세먼지 어때?", "오늘 {city} 공기 괜찮아?", "{city} 대기질 알려{e}", "{city} 초미세먼지 수치 확인해{e}"],
        "set_thermostat": ["보일러 {temp}도로 맞춰{e}", "에어컨 {temp}도로 설정해{e}", "{room} 온도 {temp}도로 올려{e}", "{room} 에어컨 {temp}도로 켜{e}", "난방 {temp}도로 해{e}"],
        "control_light": ["{room} 불 켜{e}", "{room} 조명 꺼{e}", "{room} 불 {pct}퍼센트로 줄여{e}", "{room} 조명 켜고 밝기 {pct}으로 해{e}", "{room} 불 좀 꺼{e}", "{room} 조명 {color}색으로 켜{e}"],
        "run_appliance": ["{appl} 돌려{e}", "{appl} 시작해{e}", "{n}분 뒤에 {appl} 켜{e}", "{appl} {mode2} 모드로 돌려{e}", "{appl} 지금 작동시켜{e}"],
        "lock_door": ["현관문 잠가{e}", "문 열어{e}", "도어락 잠금 해제해{e}", "현관 잠겨 있는지 잠가{e}", "뒷문 잠가{e}"],
        "set_tv": ["TV 켜{e}", "티비 꺼{e}", "TV {n}번 채널로 돌려{e}", "TV 볼륨 {pct}으로 해{e}", "TV에서 {app} 앱 실행해{e}"],
        "play_music": ["{song} 틀어{e}", "{song} 재생해{e}", "{song} 셔플로 틀어{e}", "{room} 스피커로 {song} 틀어{e}", "{song} 좀 들려{e}"],
        "media_control": ["일시정지해{e}", "다음 곡 틀어{e}", "이전 곡으로 돌아가{e}", "다시 재생해{e}", "음악 멈춰{e}", "다음 노래 틀어{e}"],
        "set_volume": ["볼륨 {pct}으로 해{e}", "소리 {pct}까지 올려{e}", "볼륨 {pct}으로 줄여{e}", "{room} 스피커 볼륨 {pct}으로 맞춰{e}"],
        "play_video": ["{drama} 틀어{e}", "{drama} {n}화 재생해{e}", "{drama} 이어서 보여{e}", "{drama} 재생해{e}"],
        "order_food": ["{food} 시켜{e}", "{food} {n}인분 배달 주문해{e}", "{food} 주문해{e}, 문 앞에 놔달라고", "{food}랑 {food2} 배달시켜{e}", "{food} {n}개 주문해{e}"],
        "search_product": ["{prod} 찾아{e}", "{money} 이하 {prod} 검색해{e}", "{prod} 인기순으로 보여{e}", "{prod} 싼 순으로 검색해{e}", "{prod} 후기 좋은 순으로 찾아{e}", "{prod} 최신순 검색해{e}"],
        "add_to_cart": ["{prod} 장바구니에 담아{e}", "{prod} {n}개 장바구니 넣어{e}", "{prod} 카트에 추가해{e}", "{prod} {n}개 담아{e}"],
        "track_delivery": ["송장번호 {track} 배송 조회해{e}", "택배 {track} 어디까지 왔어?", "{track} 배송 상태 확인해{e}", "운송장 {track} 추적해{e}"],
        "book_restaurant": ["{date} {time} {rest} {n}명 예약해{e}", "{rest} {date} {time}에 {n}명 자리 잡아{e}", "{date} {time} {rest} 예약, {n}명, 창가 자리로{e}", "{rest} {date} {time} {n}인 예약해{e}"],
        "check_balance": ["잔액 확인해{e}", "통장에 얼마 있어?", "{acct} 잔액 알려{e}", "계좌 잔고 보여{e}", "{acct} 통장 얼마 남았어?"],
        "lookup_exchange_rate": ["{cur} 환율 알려{e}", "{money2} {cur}는 원화로 얼마야?", "{cur}에서 {cur2} 환율 확인해{e}", "오늘 {cur} 환율 어때?", "{n}00 {cur} 원화로 바꾸면 얼마야?"],
        "list_transactions": ["최근 거래 내역 보여{e}", "지난 {n}일 카드 사용 내역 알려{e}", "이번 달 {acct} 입출금 내역 확인해{e}", "최근 {n}일 지출 알려{e}", "식비로 얼마 썼는지 {n}일치 보여{e}"],
        "get_stock_price": ["{stock} 주가 알려{e}", "{stock} 지금 얼마야?", "{stock} 현재가 확인해{e}", "{stock} 주식 얼마야?"],
        "find_public_service": ["가까운 {svc} 찾아{e}", "{city} {svc} 어디야?", "근처 {svc} 알려{e}", "{place} 근처 {svc} 검색해{e}"],
        "request_certificate": ["{cert} 발급 신청해{e}", "{cert} {n}통 온라인으로 발급해{e}", "{cert} 우편으로 신청해{e}", "{cert} 떼{e}", "{cert} {n}부 발급받아{e}"],
        "report_issue": ["{place} 앞 {issue} 신고해{e}", "{issue} 신고할게, 위치는 {place}", "{place}에 {issue} 있어, 신고해{e}", "{issue} 민원 넣어{e}, {place}"],
        "log_health": ["오늘 {metric} {val} 기록해{e}", "{metric} {val}로 저장해{e}", "{date} {metric} {val} 입력해{e}", "{metric} {val} 기록해{e}"],
        "book_clinic": ["{date} {clinic} 예약해{e}", "{date} {time} {clinic} {dept} 진료 예약해{e}", "{clinic} {date}로 예약 잡아{e}", "{date} {clinic} {dept} 예약해{e}"],
        "medication_reminder": ["{med} {time}에 먹으라고 알려{e}", "{med} 복약 알림 {time}, {time2}로 설정해{e}", "{n}일 동안 {time}에 {med} 알림 맞춰{e}", "{med} 알림 매일 {time}으로 설정해{e}"],
        "create_note": ["메모해{e}: {note}", "{note} 메모 남겨{e}", "{note} 라고 적어{e}", "노트에 {note} 저장해{e}", "{ntitle} 메모 만들어{e}, 내용은 {note}"],
        "add_todo": ["할 일에 {task} 추가해{e}", "{date}까지 {task} 할 일 넣어{e}", "{task} 투두에 넣어{e}", "{task} 중요 할 일로 등록해{e}", "{date} 마감으로 {task} 추가해{e}"],
        "search_notes": ["{note_q} 메모 찾아{e}", "메모에서 {note_q} 검색해{e}", "{note_q} 적어둔 거 있어?", "{note_q} 관련 노트 보여{e}"],
        "set_device_setting": ["{setting} 켜{e}", "{setting} 꺼{e}", "{setting} {pct}으로 해{e}", "{setting} 켜 {e}", "{setting} 좀 꺼{e}"],
        "take_photo": ["사진 찍어{e}", "셀카 모드로 찍어{e}", "{n}초 뒤에 사진 찍어{e}", "영상 촬영 시작해{e}", "타이머 {n}초로 사진 찍어{e}"],
        "open_app": ["{app} 열어{e}", "{app} 앱 실행해{e}", "{app} 켜{e}", "{app} 실행해{e}"],
        "translate_text": ["{phrase} {lang}로 번역해{e}", "{lang}로 {phrase} 어떻게 말해?", "{phrase}를 {lang}로 바꿔{e}", "{phrase} {lang} 번역해{e}"],
        "compose_email": ["{who}한테 {subj} 제목으로 메일 보내{e}", "{who}에게 {subj} 이메일 보내{e}, 내용은 {body}", "{subj} 메일 {who}한테 써{e}", "{who} 앞으로 {subj} 메일 발송해{e}"],
        "schedule_meeting": ["{date} {time}에 {who}랑 {title} 회의 잡아{e}", "{date} {time} {title} 회의 {n}분짜리로 잡아{e}", "{title} 회의 {date} {time} 온라인으로 잡아{e}", "{who} 포함해서 {date} {time} 회의 소집해{e}"],
        "request_leave": ["{date} {leave} 신청해{e}", "{date}부터 {date2}까지 {leave} 올려{e}", "{date} {leave} 내{e}", "{leave} {date}로 신청해{e}"],
        "convert_units": ["{n}{u1}는 몇 {u2}야?", "{n}{u1}을 {u2}로 바꿔{e}", "{n} {u1} {u2}로 변환해{e}", "{n}{u1} {u2}로 계산해{e}"],
        "web_search": ["{q} 검색해{e}", "{q} 찾아봐{e}", "{q}에 대해 알아봐{e}", "{q} 검색해{e}"],
        "get_news": ["{topic} 뉴스 보여{e}", "오늘 {topic} 뉴스 {n}개 알려{e}", "최신 뉴스 알려{e}", "{topic} 관련 헤드라인 읽어{e}"],
        "define_word": ["{word} 뜻이 뭐야?", "{word} 사전에서 찾아{e}", "{word} 의미 알려{e}", "{word} 한자로 뜻 찾아{e}"],
        "calculate": ["{expr} 계산해{e}", "{expr}은 얼마야?", "{expr} 얼마야?", "계산기: {expr}"],
    }
    return F[name]


def fill(name, frame, rng, reg):
    """Fill one frame; return (query, arguments)."""
    a = {}
    e = rng.choice(ENDINGS[reg])
    def ending(word):
        # verbs in frames end with a stem like 켜/꺼/보내/알려/찾아; ENDINGS glue after a stem that ends in 해 only.
        return word
    s = frame
    def sub(key, word):
        nonlocal s
        s = s.replace("{" + key + "}", word, 1)
    if "{who}" in s:
        w = rng.choice(NAMES); sub("who", w)
        a["recipient" if name == "send_chat_message" else "contact" if name == "make_call" else "to" if name == "compose_email" else "attendees"] = w if name != "schedule_meeting" and name != "create_event" else [w]
    if "{msg}" in s:
        m = rng.choice(["10분 늦어", "지금 출발해", "저녁 뭐 먹을래", "회의 자료 보냈어", "도착하면 연락 줘", "오늘 고마웠어", "내일 봐", "약속 장소 어디야"]); sub("msg", m); a["message"] = m
    if "{date2}" in s:
        w, d = rel_date(rng); sub("date2", w); a["end_date"] = d
    if "{date}" in s:
        w, d = rel_date(rng); sub("date", w)
        a["date" if name not in ("set_reminder", "request_leave", "add_todo") else {"set_reminder": "datetime", "request_leave": "start_date", "add_todo": "due"}[name]] = d
    if "{time2}" in s:
        w, t = rel_time(rng); sub("time2", w); a.setdefault("times", []).append(t)
    if "{time}" in s:
        w, t = rel_time(rng); sub("time", w)
        key = {"create_event": "start_time", "schedule_meeting": "start_time", "book_train": "time", "book_restaurant": "time", "book_clinic": "time",
               "estimate_travel_time": "depart_at", "set_alarm": "time", "medication_reminder": "times", "set_reminder": "datetime"}.get(name, "time")
        if key == "times":
            a.setdefault("times", []).insert(0, t)
        elif key == "datetime":
            a["datetime"] = (a.get("datetime", TODAY.isoformat()) + " " + t) if "매" not in s else a.get("datetime", TODAY.isoformat()) + " " + t
        else:
            a[key] = t
    if name == "set_reminder" and "datetime" in a and " " not in a["datetime"]:
        a["datetime"] += " 09:00"
    if name == "set_reminder" and "매일" in s:
        a["repeat"] = "daily"
    if name == "set_reminder" and "매주" in s:
        a["repeat"] = "weekly"
    if name == "set_alarm" and "평일" in s:
        a["days"] = ["mon", "tue", "wed", "thu", "fri"]
    if name == "set_alarm" and "주말" in s:
        a["days"] = ["sat", "sun"]
    if "{title}" in s:
        t = rng.choice(TITLES); sub("title", t); a["title"] = t
    if "{place2}" in s:
        p = rng.choice(PLACES); sub("place2", p); a["origin"] = p
    if "{place}" in s:
        p = rng.choice(PLACES); sub("place", p)
        a[{"navigate_to": "destination", "estimate_travel_time": "destination", "call_taxi": "destination", "get_bus_arrival": "stop_name", "create_event": "location",
           "report_issue": "location", "find_public_service": "location"}.get(name, "location")] = p
    if "{mode}" in s:
        m = rng.choice([("차로", "car"), ("대중교통으로", "transit"), ("걸어서", "walk"), ("자전거로", "bike")] if name == "navigate_to" else [("차로", "car"), ("대중교통으로", "transit"), ("걸어서", "walk")])
        sub("mode", m[0]); a["mode"] = m[1]
    if name == "navigate_to" and "톨게이트" in s:
        a["avoid_tolls"] = True
    if "{cat}" in s:
        c = rng.choice(list(CAT_KO)); sub("cat", CAT_KO[c]); a["category"] = c
        if "문 연" in s:
            a["open_now"] = True
    if "{station}" in s:
        st = rng.choice(["강남", "홍대입구", "서울", "잠실", "사당", "신촌", "왕십리", "수원", "서면", "센텀시티"]); sub("station", st); a["station"] = st
        if "상행" in s:
            a["direction"] = "up"
        if "하행" in s:
            a["direction"] = "down"
    if "{line}" in s:
        l = rng.choice(["1", "2", "3", "4", "5", "7", "9"]); sub("line", l); a["line"] = l + "호선"
    if "{city2}" in s:
        c = rng.choice(CITIES); sub("city2", c); a["destination"] = c
    if "{city}" in s:
        c = rng.choice(CITIES); sub("city", c)
        a["origin" if name == "book_train" and "destination" in a else "destination" if name == "book_train" else "location"] = c
        if name == "book_train" and "destination" not in a:
            a["destination"] = c
    if name == "book_train" and "특실" in s:
        a["seat_class"] = "first"
    if name == "call_taxi":
        if "대형" in s:
            a["vehicle"] = "large"
        if "모범" in s:
            a["vehicle"] = "premium"
    if "{world}" in s:
        w = rng.choice(WORLD); sub("world", w); a["city"] = w
    if "{temp}" in s:
        t = rng.choice([18, 20, 22, 23, 24, 25, 26, 27]); sub("temp", str(t)); a["temperature_c"] = t
        a["device"] = "boiler" if ("보일러" in s or "난방" in s) else "air_conditioner"
    if "{room}" in s:
        r = rng.choice(ROOMS); sub("room", r); a["room" if name != "set_volume" and name != "play_music" else "device"] = r if name not in ("set_volume", "play_music") else r + " 스피커"
    if name == "control_light":
        a["state"] = "off" if "꺼" in s else "on"
    if "{pct}" in s:
        p = rng.choice([10, 20, 30, 40, 50, 60, 70, 80]); sub("pct", str(p))
        a["brightness" if name == "control_light" else "level" if name == "set_volume" else "value"] = p if name != "set_device_setting" and name != "set_tv" else str(p)
    if "{color}" in s:
        c = rng.choice(["노란", "흰", "주황", "파란"]); sub("color", c); a["color"] = c + "색"
    if "{appl}" in s:
        k = rng.choice(list(APPL_KO)); sub("appl", APPL_KO[k]); a["appliance"] = k
    if "{mode2}" in s:
        m = rng.choice(["강력", "표준", "절전", "울 세탁", "조용"]); sub("mode2", m); a["mode"] = m
    if name == "lock_door":
        a["action"] = "unlock" if ("열어" in s or "해제" in s) else "lock"
        if "뒷문" in s:
            a["door"] = "뒷문"
        elif "현관" in s:
            a["door"] = "현관"
    if name == "set_tv":
        a["action"] = "power_off" if "꺼" in s else "channel" if "채널" in s else "volume" if "볼륨" in s else "open_app" if "{app}" in s else "power_on"
    if "{app}" in s:
        ap_ = rng.choice(APPS); sub("app", ap_); a["app_name" if name == "open_app" else "value"] = ap_
    if "{song}" in s:
        sg = rng.choice(SONGS); sub("song", sg); a["query"] = sg
        if "셔플" in s:
            a["shuffle"] = True
    if name == "media_control":
        a["action"] = "pause" if ("일시정지" in s or "멈춰" in s) else "next" if "다음" in s else "previous" if "이전" in s else "resume"
    if "{drama}" in s:
        d = rng.choice(DRAMAS); sub("drama", d); a["title"] = d
    if "{food2}" in s:
        f2 = rng.choice(FOODS); sub("food2", f2)
    if "{food}" in s:
        f = rng.choice(FOODS); sub("food", f); a["items"] = [f] + ([a_ for a_ in [locals().get("f2")] if a_] if "{food2}" in frame else [])
        if "문 앞" in s:
            a["delivery_note"] = "문 앞에 놓아주세요"
    if "{prod}" in s:
        p = rng.choice(PRODUCTS); sub("prod", p); a["query" if name == "search_product" else "product"] = p
        if name == "search_product":
            for k, v in (("인기순", "popular"), ("싼 순", "price_low"), ("후기 좋은", "rating"), ("최신순", "newest")):
                if k in s:
                    a["sort"] = v
    if "{money2}" in s:
        n_ = rng.choice([100, 300, 500, 1000]); sub("money2", str(n_)); a["amount"] = n_
    if "{money}" in s:
        w, v = money(rng); sub("money", w); a["max_price_krw"] = v
    if "{track}" in s:
        t = "".join(rng.choice("0123456789") for _ in range(12)); sub("track", t); a["tracking_number"] = t
    if "{rest}" in s:
        r = rng.choice(["한식당", "이탈리안 식당", "고깃집", "초밥집", "브런치 카페", "양식당"]); sub("rest", r); a["restaurant"] = r
        if "창가" in s:
            a["request"] = "창가 자리"
    if "{acct}" in s:
        ac = rng.choice(["월급 통장", "생활비 통장", "적금", "주거래"]); sub("acct", ac); a["account"] = ac
    if "{cur2}" in s:
        c2 = rng.choice([c for c in CURR if c != "KRW"]); sub("cur2", CURR_KO[c2]); a["to_currency"] = c2
    if "{cur}" in s:
        c = rng.choice([c for c in CURR if c != "KRW" and c != a.get("to_currency")]); sub("cur", CURR_KO[c]); a["from_currency"] = c
        a.setdefault("to_currency", "KRW")
    if "{stock}" in s:
        st = rng.choice(STOCKS); sub("stock", st); a["ticker_or_name"] = st
    if "{svc}" in s:
        k = rng.choice(list(SVC_KO)); sub("svc", SVC_KO[k]); a["service"] = k
    if "{cert}" in s:
        k = rng.choice(list(CERT_KO)); sub("cert", CERT_KO[k]); a["document"] = k
        if "온라인" in s:
            a["delivery"] = "online"
        if "우편" in s:
            a["delivery"] = "mail"
    if "{issue}" in s:
        k = rng.choice(list(ISSUE_KO)); sub("issue", ISSUE_KO[k]); a["issue_type"] = k
    if "{metric}" in s:
        k = rng.choice(list(METRIC_KO)); label, vals = METRIC_KO[k]; v = rng.choice(vals); sub("metric", label); a["metric"] = k
        sub("val", v); a["value"] = v
    if "{clinic}" in s:
        c = rng.choice(["동네 내과", "치과", "피부과", "정형외과", "한의원", "안과"]); sub("clinic", c); a["clinic"] = c
    if "{dept}" in s:
        d = rng.choice(["내과", "정형외과", "피부과", "소아과"]); sub("dept", d); a["department"] = d
    if "{med}" in s:
        m = rng.choice(MEDICINES); sub("med", m); a["medicine"] = m
    if "{note_q}" in s:
        q = rng.choice(["주차", "비밀번호", "발표", "여행", "장보기", "회의"]); sub("note_q", q); a["query"] = q
    if "{ntitle}" in s:
        t = rng.choice(["장보기", "아이디어", "회의 메모", "여행 준비"]); sub("ntitle", t); a["title"] = t
    if "{note}" in s:
        n_ = rng.choice(NOTE_BITS); sub("note", n_); a["content"] = n_
    if "{task}" in s:
        t = rng.choice(TASKS); sub("task", t); a["task" if name == "add_todo" else "text"] = t
        if name == "add_todo" and "중요" in s:
            a["priority"] = "high"
    if "{setting}" in s:
        k = rng.choice(list(SET_KO)); sub("setting", SET_KO[k]); a["setting"] = k
        if "value" not in a:
            a["value"] = "off" if "꺼" in s else "on"
    if name == "take_photo":
        a["mode"] = "selfie" if "셀카" in s else "video" if "영상" in s else "photo"
    if "{phrase}" in s:
        p = rng.choice(["화장실이 어디예요", "이거 얼마예요", "도와주세요", "감사합니다", "예약했어요", "천천히 말씀해 주세요"]); sub("phrase", p); a["text"] = p
    if "{lang}" in s:
        k = rng.choice([k for k in LANG_KO if k != "ko"]); sub("lang", LANG_KO[k]); a["target_language"] = k
    if "{subj}" in s:
        sj = rng.choice(DOCS); sub("subj", sj); a["subject"] = sj
    if "{body}" in s:
        b = rng.choice(["검토 부탁드립니다", "첨부 확인해 주세요", "금요일까지 회신 주세요"]); sub("body", b); a["body"] = b
    if name == "schedule_meeting" and "온라인" in s:
        a["online"] = True
    if "{leave}" in s:
        k = rng.choice(list(LEAVE_KO)); sub("leave", LEAVE_KO[k]); a["type"] = k
    if "{u1}" in s:
        u1, u2 = rng.choice(UNIT_PAIRS); sub("u1", u1); sub("u2", u2); a["from_unit"] = u1; a["to_unit"] = u2
    if "{q}" in s:
        q = rng.choice(["제주도 가을 여행지", "노트북 배터리 오래 쓰는 법", "김치찌개 레시피", "2026년 추석 연휴", "전기차 보조금 조건", "초보 러닝 루틴"]); sub("q", q); a["query"] = q
    if "{topic}" in s:
        t = rng.choice(["경제", "IT", "스포츠", "날씨", "부동산", "연예"]); sub("topic", t); a["topic"] = t
    if "{word}" in s:
        w = rng.choice(WORDS); sub("word", w); a["word"] = w
        a["dictionary"] = "hanja" if "한자" in s else ("english" if w.isascii() else "korean")
    if "{expr}" in s:
        x = rng.choice(EXPR); sub("expr", x); a["expression"] = x
    if "{label}" in s:
        l = rng.choice(["출근", "운동", "약", "라면", "빨래"]); sub("label", l); a["label"] = l
    if "{n}" in s:
        lo, hi = {"set_timer": (1, 60), "read_unread_messages": (1, 10), "find_nearby": (2, 9), "get_bus_arrival": (1, 999), "book_train": (1, 6), "order_food": (1, 4),
                  "add_to_cart": (1, 5), "book_restaurant": (2, 8), "list_transactions": (3, 30), "request_certificate": (1, 3), "medication_reminder": (3, 30), "take_photo": (3, 10),
                  "get_news": (3, 10), "convert_units": (1, 500), "schedule_meeting": (15, 90), "play_video": (1, 16), "set_tv": (1, 99), "run_appliance": (10, 120), "lookup_exchange_rate": (1, 9)}.get(name, (1, 20))
        w, v = num(rng, lo, hi); sub("n", w)
        key = {"set_timer": "seconds" if "초" in s else "minutes", "read_unread_messages": "limit", "find_nearby": "radius_m", "get_bus_arrival": "route", "book_train": "passengers",
               "order_food": "quantity", "add_to_cart": "quantity", "book_restaurant": "party_size", "list_transactions": "days", "request_certificate": "copies",
               "medication_reminder": "days", "take_photo": "timer_seconds", "get_news": "count", "convert_units": "value", "schedule_meeting": "duration_minutes",
               "play_video": "episode", "set_tv": "value", "run_appliance": "delay_minutes", "lookup_exchange_rate": "amount"}.get(name)
        if key:
            a[key] = (v * 100 if name == "lookup_exchange_rate" else v * 100 if name == "find_nearby" else str(v) if key in ("route", "value") and name in ("get_bus_arrival", "set_tv") else v)
            if name == "find_nearby":
                s = s.replace(f"{w}미터", f"{v*100}미터")
    if name == "set_tv" and a.get("action") in ("channel", "volume") and "value" in a and not isinstance(a["value"], str):
        a["value"] = str(a["value"])
    if name == "search_product" and "query" not in a:
        return None
    s = s.replace("{e}", e)
    # typo / spacing noise for a slice of examples
    if rng.random() < 0.12:
        s = s.replace(" ", "", 1) if rng.random() < 0.5 else s.replace("요", "용", 1)
    if rng.random() < 0.08:
        s = s.replace("해 주세요", "해주세요").replace("주세요", "주세용")
    if rng.random() < 0.08:  # 경상도/전라도 어미
        s = s.replace("해줘", rng.choice(["해도", "해주라", "해주소"])).replace("어때?", rng.choice(["어떻노?", "어떻당가?", "어떤데?"]))
    return s, a


NEGATIVES = [
    "오늘 하루 어땠어?", "심심한데 재밌는 얘기 해줘", "너 이름이 뭐야?", "요즘 왜 이렇게 피곤하지", "점심 뭐 먹을지 고민이야", "고마워", "잘 자", "나 오늘 기분 좋아",
    "사람은 왜 꿈을 꿀까", "우주는 얼마나 클까", "한국에서 제일 높은 산이 어디야", "김치는 왜 매워", "넌 감정이 있어?", "농담 하나 해봐", "안녕", "뭐해?", "배고파",
    "오늘 회의 진짜 길었어", "요즘 운동 시작했는데 힘들다", "책 추천해줄 수 있어?", "인생 조언 좀 해줘", "네가 좋아하는 색은 뭐야", "바다랑 산 중에 뭐가 좋아", "졸려",
    "시험 망했어", "주말에 뭐 하지", "커피 너무 많이 마신 것 같아", "우리 강아지가 아파", "비 오는 날은 왜 우울할까", "칭찬 좀 해줘", "이 노래 가사가 무슨 뜻일까",
    "예전에 살던 동네가 그리워", "세상에서 제일 맛있는 음식은 뭘까", "너는 어디에 있어?", "옛날 이야기 해줘", "하늘은 왜 파래", "한글날이 언제더라, 그냥 궁금해서",
]
NEG_ENDINGS = ["", "?", "..", " ㅋㅋ", " ㅠㅠ", "~", " 진짜", " 그냥 궁금해서"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60000); ap.add_argument("--neg", type=int, default=4000)
    ap.add_argument("--out", default=str(ROOT / "data" / "ko_raw.jsonl")); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(); rng = random.Random(a.seed)
    names = list(TOOLS); seen = set(); rows = []
    tries = 0
    while len(rows) < a.n and tries < a.n * 8:
        tries += 1
        name = rng.choice(names); frame = rng.choice(frames(name)); reg = rng.choices(["banmal", "jondae", "formal"], [0.45, 0.4, 0.15])[0]
        r = fill(name, frame, rng, reg)
        if not r:
            continue
        q, args = r
        if "{" in q or q in seen:
            continue
        props = TOOLS[name]["parameters"].get("properties", {}); req = TOOLS[name]["parameters"].get("required", [])
        args = {k: v for k, v in args.items() if k in props}
        if any(k not in args for k in req):
            continue
        for k, v in args.items():
            if "enum" in props[k] and v not in props[k]["enum"]:
                break
        else:
            seen.add(q); rows.append({"tool": name, "query": q, "call": {"name": name, "arguments": args}})
    for _ in range(a.neg):
        q = rng.choice(NEGATIVES) + rng.choice(NEG_ENDINGS)
        rows.append({"tool": None, "query": q, "call": {"name": "none", "arguments": {}}})
    rng.shuffle(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    per = {}
    for r in rows:
        per[r["tool"]] = per.get(r["tool"], 0) + 1
    print(f"wrote {a.out}: {len(rows)} rows, {len(per)} tools, min/tool {min(v for k, v in per.items() if k)} max {max(per.values())}")
    for r in rows[:6]:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
