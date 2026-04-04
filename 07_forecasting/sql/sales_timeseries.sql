-- sales_timeseries.sql — Serie temporal para forecasting

SELECT
    DATE_TRUNC('month', sale_date)   AS ds,
    SUM(revenue)                      AS y,
    COUNT(DISTINCT order_id)          AS orders,
    ROUND(AVG(revenue), 2)            AS avg_ticket
FROM sales
WHERE sale_date >= CURRENT_DATE - INTERVAL '3 years'
  AND status = 'completed'
GROUP BY 1
ORDER BY 1;
