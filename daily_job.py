import sqlite3
import yfinance as yf
import requests
from google import genai
import time
from datetime import datetime

# app.py에서 기존 핵심 로직 및 변수들을 임포트합니다.
from app import get_stock_news, DB_FILE, TELEGRAM_TOKEN, GEMINI_API_KEY

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def run_daily_job():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 알파 터미널 모닝 브리핑 자동화 배치 작업을 시작합니다.")
    
    # 1. 최신 환율 가져오기 (원화 환산용)
    try:
        fx_hist = yf.Ticker("KRW=X").history(period='1d')
        exchange_rate = float(fx_hist['Close'].iloc[-1])
        print(f"✅ 현재 환율 로드 완료: 1$ = ₩{exchange_rate:.2f}")
    except:
        exchange_rate = 1400.0
        print(f"⚠️ 환율 로드 실패. 기본값 ₩{exchange_rate:.2f} 적용")

    conn = get_db_connection()
    # 텔레그램 ID를 등록한 유저들만 조회
    users = conn.execute("SELECT id, username, telegram_chat_id FROM users WHERE telegram_chat_id IS NOT NULL AND telegram_chat_id != ''").fetchall()
    print(f"✅ 알림 수신 대상 유저: 총 {len(users)}명")
    
    client = genai.Client(api_key=GEMINI_API_KEY)

    for user in users:
        user_id = user['id']
        username = user['username']
        chat_id = user['telegram_chat_id']
        
        print(f"🔄 [{username}]님의 데이터를 분석 중...")
        
        # 유저별 포트폴리오 로드
        rows = conn.execute('SELECT * FROM portfolios WHERE user_id = ?', (user_id,)).fetchall()
        portfolio = [dict(row) for row in rows]
        
        if not portfolio: 
            print(f"⏩ [{username}]님은 포트폴리오가 비어있어 건너뜁니다.")
            continue

        total_current_usd = 0
        total_principal_usd = 0
        
        # 최신 주가로 자산 정산
        for item in portfolio:
            ticker = item['ticker']
            shares = item['shares']
            avg_price = item['avgPrice']
            market = item['market']
            
            # 현재가 조회 (한국 주식은 .KQ 폴백 지원)
            try:
                hist = yf.Ticker(ticker).history(period='1d')
                if hist.empty and ticker.endswith('.KS'):
                    hist = yf.Ticker(ticker.replace('.KS', '.KQ')).history(period='1d')
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                else:
                    current_price = avg_price
            except:
                current_price = avg_price
            
            item['currentPrice'] = current_price
            
            current_val = current_price * shares
            principal_val = avg_price * shares
            
            if market == 'KR':
                total_current_usd += current_val / exchange_rate
                total_principal_usd += principal_val / exchange_rate
            else:
                total_current_usd += current_val
                total_principal_usd += principal_val
                
        # 총 수익률 계산
        if total_principal_usd > 0:
            total_return_pct = ((total_current_usd - total_principal_usd) / total_principal_usd) * 100
        else:
            total_return_pct = 0.0
            
        portfolio_summary = f"총 잔고: ${total_current_usd:,.2f} / 총 수익률: {total_return_pct:+.2f}%"
        
        # 3. 맞춤형 뉴스 수집
        print(f"📰 [{username}]님의 보유 종목 뉴스를 스크랩합니다...")
        stock_news = get_stock_news(portfolio)
        news_titles = [news['title'] for news in stock_news]
        
        # 4. AI 브리핑 프롬프트 조립 (고도화된 리포트)
        prompt = f"""
        당신은 냉철하고 분석적인 월스트리트의 수석 퀀트 투자 애널리스트 '알파 엔진'입니다.
        아래 제공된 [내 계좌 상태]와 [실시간 뉴스 헤드라인]을 심층적으로 분석하여, 다음 세 가지 섹션으로 구성된 **전문적이고 상세한 일일 투자 리포트**를 작성해 주세요.
        
        [작성 가이드]
        1. 📊 [마켓 오버뷰]: 실시간 뉴스를 기반으로 현재 글로벌 거시 경제 및 주요 시장의 흐름과 핵심 이슈를 요약하세요.
        2. 💼 [내 계좌 진단]: 내 계좌의 현재 상태(총 잔고, 수익률)를 평가하고, 거시 경제 상황이 내 포트폴리오에 미칠 영향을 분석하세요.
        3. 🎯 [오늘의 액션 플랜]: 관망, 추매, 헷징 등 오늘 당장 취해야 할 가장 합리적이고 구체적인 전략을 제시하세요.
        
        말투는 투자자에게 확신과 경각심을 동시에 주는 단호하고 전문적인 문어체(예: ~할 것으로 보입니다, ~해야 합니다)를 사용하십시오. 이모지를 적절히 활용하여 시각적으로 읽기 편하게 구성하세요.

        [내 계좌 상태]
        {portfolio_summary}

        [실시간 뉴스 헤드라인]
        {chr(10).join('- ' + t for t in news_titles[:15]) if news_titles else '현재 수집된 실시간 뉴스가 없습니다.'}
        """
        
        print(f"🧠 [{username}]님의 AI 리포트 생성 중...")
        try:
            response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            ai_summary = response.text.strip()
        except Exception as e:
            ai_summary = "⚠️ AI 분석 엔진 구동 중 오류가 발생했습니다."
            print(f"❌ AI 분석 에러: {e}")
            
        # 5. 텔레그램 메세지 발송
        try:
            if TELEGRAM_TOKEN != "여기에_텔레그램_봇_토큰을_넣으세요":
                tg_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
                message_text = f"🌅 [JURUNG QUANT 모닝 브리핑]\n\n{portfolio_summary}\n\n🧠 [알파 엔진의 일일 리포트]\n{ai_summary}"
                
                # 마크다운 충돌 방지를 위해 파싱 모드는 생략
                res = requests.post(tg_url, data={'chat_id': chat_id, 'text': message_text})
                if res.status_code == 200:
                    print(f"📨 [{username}]님에게 텔레그램 발송 성공!")
                else:
                    print(f"❌ [{username}]님 텔레그램 발송 실패: {res.text}")
        except Exception as e:
            print(f"❌ 텔레그램 통신 에러: {e}")
            
        # 과도한 API 호출 방지를 위한 휴식
        time.sleep(2)
            
    conn.close()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🎉 알파 터미널 자동화 배치 작업 완료!")

if __name__ == "__main__":
    run_daily_job()
