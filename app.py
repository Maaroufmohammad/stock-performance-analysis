"""A small Streamlit app for viewing adjusted stock closing prices."""

from datetime import date, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="Stock Prices", page_icon="📈")


@st.cache_data
def download_adjusted_prices(
    tickers: tuple[str, ...], start_date: date, end_date: date
) -> pd.DataFrame:
    """Download and cache adjusted closing prices for the requested period."""
    # Include extra history for the 200-day moving average. yfinance's end
    # date is exclusive, so add one day to include the user's selected end date.
    downloaded = yf.download(
        tickers=list(tickers),
        start=start_date - timedelta(days=300),
        end=end_date + timedelta(days=1),
        auto_adjust=False,
        progress=False,
    )

    if downloaded.empty:
        return pd.DataFrame()

    # yfinance may return regular or MultiIndex columns depending on the
    # number of tickers and its version.
    if isinstance(downloaded.columns, pd.MultiIndex):
        adjusted_prices = None
        for level in range(downloaded.columns.nlevels):
            if "Adj Close" in downloaded.columns.get_level_values(level):
                adjusted_prices = downloaded.xs("Adj Close", axis=1, level=level)
                break
        if adjusted_prices is None:
            return pd.DataFrame()
    elif "Adj Close" in downloaded.columns:
        adjusted_prices = downloaded[["Adj Close"]].rename(
            columns={"Adj Close": tickers[0]}
        )
    else:
        return pd.DataFrame()

    if isinstance(adjusted_prices, pd.Series):
        adjusted_prices = adjusted_prices.to_frame(name=tickers[0])

    # Preserve the selected ticker order and provide a consistent set of columns.
    return adjusted_prices.reindex(columns=list(tickers))


st.title("Adjusted Stock Closing Prices")
st.write("Choose one or more tickers and a date range to view historical prices.")

# These are the example stocks and market index available in the sidebar.
available_tickers = ["AAPL", "JPM", "XOM", "TSLA", "JNJ", "SPY"]

with st.sidebar:
    st.header("Choose your data")
    selected_tickers = st.multiselect(
        "Stock tickers",
        options=available_tickers,
        default=["AAPL", "JPM", "XOM", "TSLA", "JNJ", "SPY"],
    )
    if selected_tickers:
        moving_average_ticker = st.selectbox(
            "Ticker for moving averages",
            options=selected_tickers,
        )

    today = date.today()
    start_date = st.date_input(
        "Start date",
        value=today - timedelta(days=365),
        max_value=today,
    )
    end_date = st.date_input(
        "End date",
        value=today,
        max_value=today,
    )

if not selected_tickers:
    st.info("Select at least one ticker in the sidebar.")
elif start_date > end_date:
    st.error("The start date must be on or before the end date.")
else:
    with st.spinner("Downloading adjusted closing prices..."):
        try:
            adjusted_prices = download_adjusted_prices(
                tuple(selected_tickers), start_date, end_date
            )

            if adjusted_prices.empty:
                st.warning(
                    "No price data was returned for that ticker/date selection. "
                    "Check the tickers and try a different date range."
                )
            else:
                # The downloaded data includes the extra history for calculating
                # moving averages. Show the table and returns only for the chosen
                # date range.
                selected_period_prices = adjusted_prices.loc[
                    (adjusted_prices.index >= pd.Timestamp(start_date))
                    & (adjusted_prices.index <= pd.Timestamp(end_date))
                ]

                st.subheader("First 5 rows")
                st.dataframe(
                    selected_period_prices.head(5), use_container_width=True
                )

                # Start each stock's cumulative return at zero on its first
                # available date in the selected period.
                return_prices = selected_period_prices.dropna(
                    axis=1, how="all"
                )
                if return_prices.empty:
                    st.warning("No prices are available in the selected date range.")
                else:
                    first_prices = return_prices.apply(
                        lambda prices: prices.dropna().iloc[0]
                    )
                    cumulative_returns = (
                        return_prices.divide(first_prices).subtract(1).multiply(100)
                    )
                    st.subheader("Cumulative returns")
                    st.line_chart(
                        cumulative_returns,
                        x_label="Date",
                        y_label="Cumulative return (%)",
                        use_container_width=True,
                    )

                    # Calculate rolling averages on all downloaded history,
                    # then limit the chart to the selected date range.
                    stock_prices = adjusted_prices[moving_average_ticker]
                    moving_averages = pd.DataFrame(
                        {
                            "Adjusted close": stock_prices,
                            "50-day moving average": stock_prices.rolling(
                                window=50
                            ).mean(),
                            "200-day moving average": stock_prices.rolling(
                                window=200
                            ).mean(),
                        }
                    )
                    moving_averages = moving_averages.loc[
                        (moving_averages.index >= pd.Timestamp(start_date))
                        & (moving_averages.index <= pd.Timestamp(end_date))
                    ]

                    st.subheader(f"{moving_average_ticker} moving averages")
                    st.line_chart(
                        moving_averages,
                        x_label="Date",
                        y_label="Adjusted closing price (USD)",
                        use_container_width=True,
                    )

                # Compare how each pair of tickers moved using daily returns.
                daily_returns = selected_period_prices.pct_change(
                    fill_method=None
                ).dropna(axis=1, how="all")
                st.subheader("Daily return correlation")
                if daily_returns.empty or len(daily_returns) < 2:
                    st.info(
                        "At least two daily returns are needed to calculate "
                        "the correlation heatmap."
                    )
                else:
                    return_correlations = daily_returns.corr()
                    st.dataframe(
                        return_correlations.style.background_gradient(
                            cmap="RdYlGn", vmin=-1, vmax=1
                        ).format("{:.2f}", na_rep="N/A"),
                        use_container_width=True,
                    )
                    st.caption(
                        "Each cell shows the correlation between two tickers' "
                        "daily adjusted-price returns. Values near +1 indicate "
                        "they tended to move together, values near -1 indicate "
                        "they tended to move in opposite directions, and values "
                        "near 0 indicate a weaker linear relationship. "
                        "Correlation does not imply causation."
                    )

                # Calculate performance metrics only from prices in the
                # selected date range, not the extra history used above.
                trading_days_per_year = 252
                metric_rows = []
                for ticker in selected_tickers:
                    prices = selected_period_prices[ticker].dropna()
                    daily_returns = prices.pct_change().dropna()

                    annualized_return = float("nan")
                    annualized_volatility = float("nan")
                    sharpe_ratio = float("nan")
                    max_drawdown = float("nan")

                    if len(prices) >= 2 and prices.iloc[0] > 0:
                        # Compound the first-to-last return to a 252-trading-day year.
                        annualized_return = (
                            (prices.iloc[-1] / prices.iloc[0])
                            ** (trading_days_per_year / len(daily_returns))
                        ) - 1
                        max_drawdown = (
                            prices / prices.cummax() - 1
                        ).min()

                    if len(daily_returns) >= 2:
                        daily_volatility = daily_returns.std()
                        annualized_volatility = (
                            daily_volatility * trading_days_per_year**0.5
                        )

                        # Assume a risk-free rate of zero for this simple Sharpe ratio.
                        if daily_volatility > 0:
                            sharpe_ratio = (
                                daily_returns.mean()
                                / daily_volatility
                                * trading_days_per_year**0.5
                            )

                    metric_rows.append(
                        {
                            "Ticker": ticker,
                            "Annualized return": annualized_return,
                            "Annualized volatility": annualized_volatility,
                            "Sharpe ratio": sharpe_ratio,
                            "Maximum drawdown": max_drawdown,
                        }
                    )

                st.subheader("Performance metrics")
                metrics = pd.DataFrame(metric_rows).set_index("Ticker")
                st.dataframe(
                    metrics.style.format(
                        {
                            "Annualized return": "{:.2%}",
                            "Annualized volatility": "{:.2%}",
                            "Sharpe ratio": "{:.2f}",
                            "Maximum drawdown": "{:.2%}",
                        },
                        na_rep="N/A",
                    ),
                    use_container_width=True,
                )

                with st.expander("How these metrics are calculated"):
                    st.markdown(
                        """
                        Metrics use adjusted closing prices within the selected
                        date range. **N/A** means there was not enough valid
                        price data to calculate that metric.

                        - **Annualized return:** The first-to-last price return
                          compounded to a 252-trading-day year:
                          `(last price / first price) ** (252 / daily return count) - 1`.
                        - **Annualized volatility:** The sample standard
                          deviation of daily percentage returns multiplied by
                          `sqrt(252)`.
                        - **Sharpe ratio:** The mean daily return divided by
                          the sample standard deviation of daily returns,
                          multiplied by `sqrt(252)`. This assumes a **0% risk-free
                          rate**; higher values indicate more return per unit
                          of volatility.
                        - **Maximum drawdown:** The largest peak-to-trough
                          decline in the period, calculated as the minimum of
                          `price / running peak - 1`. It is shown as a negative
                          percentage (or 0% if there was no decline).
                        """
                    )
        except Exception as error:
            st.error(
                "We couldn't download the stock data. Check your internet "
                "connection and selected dates, then try again."
            )
            with st.expander("Technical details"):
                st.code(str(error))
