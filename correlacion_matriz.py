import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, date
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from fake_useragent import UserAgent
import re
import dcor

# Constants
CLOSE_COLUMN = 'Close'
MAX_TICKERS = 13
DEFAULT_START_DATE = pd.to_datetime("2023-01-01")
DEFAULT_END_DATE = pd.to_datetime(datetime.today().date())  # Today’s date
CACHE_TTL = 86400  # 24 hours

st.set_page_config(layout="wide")
st.title("Matriz de Correlación de Acciones")

# Initialize User-Agent
ua = UserAgent()

# Generic API fetch function
def fetch_data_from_api(url, params, cookies, headers, parse_func, source_name, ticker):
    headers['User-Agent'] = ua.random
    st.write(f"Debug: Fetching URL: {url} with params: {params} for {ticker} from {source_name}")
    try:
        import certifi
        response = requests.get(url, params=params, cookies=cookies, headers=headers, verify=certifi.where())
        if response.status_code == 200:
            return parse_func(response.json())
        else:
            st.error(f"Error fetching {ticker} from {source_name}: Status code {response.status_code}")
            return pd.DataFrame()
    except requests.exceptions.SSLError as e:
        st.error(f"SSL error fetching {ticker} from {source_name}: {e}. Check certificate setup.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching {ticker} from {source_name}: {e}. Check your connection.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching {ticker} from {source_name}: {e}. Check your connection.")
        return pd.DataFrame()

# Data source functions
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def descargar_datos_yfinance(tickers, start, end, resolution='1d'):
    try:
        st.write(f"Debug: Fetching {tickers} from {start} to {end} with resolution {resolution}")
        data = yf.download(tickers=' '.join(tickers), start=start, end=end, interval=resolution, group_by='ticker')
        st.write(f"Debug: yfinance returned data for {tickers} with shape {data.shape}")
        if data.empty:
            st.warning(f"No data returned by yfinance for {tickers}. Check date range or ticker validity.")
        if isinstance(data.columns, pd.MultiIndex):
            return {ticker: data[ticker] for ticker in tickers if ticker in data.columns}
        else:
            return {tickers[0]: data} if not data.empty else {}
    except Exception as e:
        st.error(f"Error downloading from yfinance: {e}")
        return {}

def parse_analisistecnico(data):
    if not all(key in data for key in ['t', 'c', 'o', 'h', 'l', 'v']):
        st.error("Incomplete data received")
        return pd.DataFrame()
    df = pd.DataFrame({
        'Date': pd.to_datetime(data['t'], unit='s'),
        CLOSE_COLUMN: data['c'],
        'Open': data['o'], 'High': data['h'], 'Low': data['l'], 'Volume': data['v']
    }).sort_values('Date').drop_duplicates(subset=['Date']).set_index('Date')
    return df[[CLOSE_COLUMN]]

def descargar_datos_analisistecnico(ticker, start_date, end_date):
    from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
    to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())
    cookies = {'ChyrpSession': '0e2b2109d60de6da45154b542afb5768', 'i18next': 'es', 'PHPSESSID': '5b8da4e0d96ab5149f4973232931f033'}
    headers = {'accept': '*/*', 'content-type': 'text/plain', 'dnt': '1', 'referer': 'https://analisistecnico.com.ar/'}
    params = {'symbol': ticker.replace('.BA', ''), 'resolution': 'D', 'from': str(from_timestamp), 'to': str(to_timestamp)}
    return fetch_data_from_api('https://analisistecnico.com.ar/services/datafeed/history', params, cookies, headers, parse_analisistecnico, 'AnálisisTécnico.com.ar', ticker)

def parse_iol(data):
    if data.get('status') != 'ok' or 'bars' not in data:
        st.error("Invalid API response")
        return pd.DataFrame()
    df = pd.DataFrame(data['bars']).assign(Date=lambda x: pd.to_datetime(x['time'], unit='s'), Close=lambda x: x['close'])[['Date', CLOSE_COLUMN]]
    return df.sort_index().drop_duplicates().set_index('Date')

def descargar_datos_iol(ticker, start_date, end_date):
    from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
    to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())
    cookies = {'intencionApertura': '0', '__RequestVerificationToken': 'DTGdEz0miQYq1kY8y4XItWgHI9HrWQwXms6xnwndhugh0_zJxYQvnLiJxNk4b14NmVEmYGhdfSCCh8wuR0ZhVQ-oJzo1', 'isLogged': '1', 'uid': '1107644'}
    headers = {'accept': '*/*', 'content-type': 'text/plain', 'referer': 'https://iol.invertironline.com'}
    params = {'symbolName': ticker.replace('.BA', ''), 'exchange': 'BCBA', 'from': str(from_timestamp), 'to': str(to_timestamp), 'resolution': 'D'}
    return fetch_data_from_api('https://iol.invertironline.com/api/cotizaciones/history', params, cookies, headers, parse_iol, 'IOL (Invertir Online)', ticker)

def parse_byma(data):
    if not all(key in data for key in ['t', 'c']):
        st.error("Incomplete data received")
        return pd.DataFrame()
    df = pd.DataFrame({'Date': pd.to_datetime(data['t'], unit='s'), CLOSE_COLUMN: data['c']}).sort_values('Date').drop_duplicates(subset=['Date']).set_index('Date')
    return df

def descargar_datos_byma(ticker, start_date, end_date):
    from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
    to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())
    cookies = {'JSESSIONID': '5080400C87813D22F6CAF0D3F2D70338', '_fbp': 'fb.2.1728347943669.954945632708052302'}
    headers = {'Accept': 'application/json, text/plain, */*', 'Referer': 'https://open.bymadata.com.ar/'}
    symbol = ticker.replace('.BA', '') + (' 24HS' if not ticker.endswith(' 24HS') else '')
    params = {'symbol': symbol, 'resolution': 'D', 'from': str(from_timestamp), 'to': str(to_timestamp)}
    return fetch_data_from_api('https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free/chart/historical-series/history', params, cookies, headers, parse_byma, 'ByMA Data', ticker)

# Remove or adjust caching
@st.cache_data(ttl=CACHE_TTL, hash_funcs={date: lambda x: x.isoformat()})
# Alternatively, use: st.cache_data.clear() in main()
def fetch_stock_data(ticker, start_date, end_date, source='YFinance', resolution='1d'):
    st.write(f"Debug: Fetching {ticker} from source {source}")
    start_date = start_date if isinstance(start_date, date) else datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = end_date if isinstance(end_date, date) else datetime.strptime(end_date, '%Y-%m-%d').date()
    if source == 'YFinance':
        return descargar_datos_yfinance([ticker], start_date, end_date, resolution).get(ticker, pd.DataFrame())
    elif source == 'AnálisisTécnico.com.ar':
        return descargar_datos_analisistecnico(ticker, start_date, end_date)
    elif source == 'IOL (Invertir Online)':
        return descargar_datos_iol(ticker, start_date, end_date)
    elif source == 'ByMA Data':
        return descargar_datos_byma(ticker, start_date, end_date)
    st.error(f"Unknown data source: {source}")
    return pd.DataFrame()

def calculate_ticker_ratio(data1, data2):
    if data1.empty or data2.empty:
        raise ValueError("One or both datasets are empty")
    common_dates = data1.index.intersection(data2.index)
    if not common_dates.size:
        raise ValueError("No overlapping dates")
    close1 = data1.reindex(common_dates)[CLOSE_COLUMN]
    close2 = data2.reindex(common_dates)[CLOSE_COLUMN]
    return pd.DataFrame({CLOSE_COLUMN: close1 / close2}, index=common_dates)

def prepare_correlation_data(tickers, start_date, end_date, source, ma_periods, ma_type, resolution='1d'):
    all_data = pd.DataFrame()
    if source == 'YFinance':
        data_dict = descargar_datos_yfinance([t for t in tickers if '/' not in t], start_date, end_date, resolution)
    else:
        data_dict = {t: fetch_stock_data(t, start_date, end_date, source, resolution) for t in tickers if '/' not in t}

    for ticker in tickers:
        if '/' in ticker:
            ticker1, ticker2 = [t.strip() for t in ticker.split('/')]
            data1 = data_dict.get(ticker1, fetch_stock_data(ticker1, start_date, end_date, source, resolution))
            data2 = data_dict.get(ticker2, fetch_stock_data(ticker2, start_date, end_date, source, resolution))
            try:
                stock_data = calculate_ticker_ratio(data1, data2)
            except ValueError as e:
                st.error(f"Skipping {ticker}: {e}")
                continue
        else:
            stock_data = data_dict.get(ticker, pd.DataFrame())
            if stock_data.empty:
                st.error(f"No data for {ticker}. Skipping.")
                continue
        
        close_prices = stock_data[CLOSE_COLUMN]
        # Compute returns
        close_prices = close_prices.pct_change().dropna()
        if ma_periods > 1:
            if ma_type == "Exponencial":
                close_prices = close_prices.ewm(span=ma_periods).mean()
            else:
                close_prices = close_prices.rolling(window=ma_periods).mean()
        all_data[ticker] = close_prices

    # Align data to common dates
    all_data = all_data.dropna(how='any')  # Keep only rows where all tickers have data
    if all_data.empty:
        st.error("No overlapping data available for the selected tickers after alignment.")
    return all_data

def plot_correlation_matrix(data, title, method='pearson'):
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    if method == 'distance':
        corr_matrix = pd.DataFrame(index=data.columns, columns=data.columns)
        for col1 in data.columns:
            for col2 in data.columns:
                # Since data is aligned, no need for individual .dropna()
                try:
                    corr_matrix.loc[col1, col2] = dcor.distance_correlation(data[col1].values, data[col2].values)
                except Exception as e:
                    st.warning(f"Failed to compute distance correlation for {col1} vs {col2}: {e}")
                    corr_matrix.loc[col1, col2] = np.nan
        corr_matrix = corr_matrix.astype(float)
        vmin, vmax = 0, 1
    else:
        corr_matrix = data.corr(method=method)
        vmin, vmax = -1, 1
    custom_cmap = sns.diverging_palette(h_neg=10, h_pos=130, s=99, l=55, sep=3, as_cmap=True)
    sns.heatmap(corr_matrix, cmap=custom_cmap, center=0.8 if method == 'distance' else 0, 
                vmin=0.7 if method == 'distance' else vmin, vmax=vmax, 
                annot=True, fmt='.2f', annot_kws={'size': 13}, 
                cbar_kws={'label': 'Correlación'}, square=True, ax=ax)
    plt.title(title, pad=20, fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.tick_params(top=True, labeltop=True, bottom=True, labelbottom=True, right=True, labelright=True)
    fig.text(0.5, 0.5, "MTaurus - X: @MTaurus_ok", fontsize=12, color='gray', ha='center', va='center', alpha=0.5)
    plt.tight_layout()
    return fig

def validate_tickers(tickers):
    return all(re.match(r'^[A-Za-z0-9./=]+$', t) for t in tickers)

def main():
    data_sources = ['YFinance', 'AnálisisTécnico.com.ar', 'IOL (Invertir Online)', 'ByMA Data']
    correlation_methods = ['pearson', 'spearman', 'kendall', 'distance']
    ma_types = ["Simple", "Exponencial"]
    resolutions = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}

    with st.sidebar:
        selected_source = st.selectbox('Seleccionar Fuente de Datos', data_sources)
        selected_method = st.selectbox('Seleccionar Método de Correlación', correlation_methods)
        with st.expander("Ayuda sobre Métodos de Correlación"):
            st.markdown("""
            **Pearson**: Mide relaciones lineales. Útil para movimientos proporcionales, pero sensible a valores extremos.
            **Spearman**: Evalúa relaciones monótonas basadas en rangos. Bueno para tendencias no lineales.
            **Kendall**: Similar a Spearman, pero más robusto para muestras pequeñas y menos sensible a outliers.
            **Distance**: Captura dependencias lineales y no lineales. Ideal para detectar patrones complejos y diferencias sutiles entre activos que se mueven de manera similar.
            """)
        ma_periods = st.number_input("Periodos de Media Móvil", min_value=1, value=5, step=1)
        ma_type = st.selectbox("Tipo de Media Móvil", ma_types)
        selected_resolution = st.selectbox("Resolución Temporal", resolutions.keys(), index=0)
        if selected_source != 'YFinance':
            st.warning("Weekly and Monthly resolutions are only supported for YFinance.")
        tickers_input = st.text_input("Ingrese hasta 13 Tickers o Relaciones (ej: AAPL, MSFT, AAPL/MSFT)", 
                                      value="AAPL, MSFT, GOOG, TSLA, AMZN, NVDA, META, UNH, JPM, V, XOM")
        start_date = st.date_input("Fecha de Inicio", value=DEFAULT_START_DATE, min_value=pd.to_datetime("1980-01-01"))
        end_date = st.date_input("Fecha de Fin", value=DEFAULT_END_DATE, max_value=datetime.today().date())
        show_data = st.checkbox("Mostrar Datos Crudos")
        confirm_data = st.button("Confirmar Datos")

    if confirm_data:
        st.cache_data.clear()  # Ensure cache doesn’t interfere
        tickers = [t.strip() for t in tickers_input.split(",")]
        if len(tickers) > MAX_TICKERS:
            st.error(f"Máximo {MAX_TICKERS} tickers permitidos.")
        elif not validate_tickers(tickers):
            st.error("Tickers inválidos. Use solo letras, números, puntos, barras o signos igual.")
        else:
            with st.spinner('Obteniendo y procesando datos...'):
                resolution = resolutions[selected_resolution]
                correlation_data = prepare_correlation_data(tickers, start_date, end_date, selected_source, ma_periods, ma_type, resolution)
                if not correlation_data.empty:
                    if show_data:
                        st.subheader("Datos Crudos")
                        st.dataframe(correlation_data)
                    fig = plot_correlation_matrix(correlation_data, 
                                                 f'Matriz de Correlación ({selected_method.capitalize()}) desde {start_date} hasta {end_date} ({selected_resolution})', 
                                                 method=selected_method)
                    st.pyplot(fig, dpi=300)
                    if st.checkbox("Mostrar Matriz Numérica"):
                        st.write(correlation_data.corr(method=selected_method))
                else:
                    st.error("No data available for the selected tickers and date range. Verify ticker symbols, dates, and resolution.")

if __name__ == "__main__":
    main()
