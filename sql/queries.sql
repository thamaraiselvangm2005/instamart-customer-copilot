-- Q1: WHERE then HAVING
-- Completed orders above INR 500 (WHERE), then customers whose total of those exceeds 8000 (HAVING)
SELECT customer_id,
       SUM(order_amount) AS total_amount
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2: three-table join
-- Completed revenue by city and product category
SELECT c.city,
       p.category,
       ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products  p ON o.product_id  = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- Q3: scalar subquery
-- Products priced above the average catalogue price
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: LEFT JOIN
-- Customers with NO completed order (the join keeps them, the order columns come back empty)
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o
       ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: common table expressions
-- Customers whose total completed spend is above the average customer total
WITH customer_totals AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_total AS (
    SELECT ROUND(AVG(total_spend), 2) AS avg_spend
    FROM customer_totals
)
SELECT t.customer_id,
       ROUND(t.total_spend, 2) AS total_spend,
       a.avg_spend
FROM customer_totals t
CROSS JOIN average_total a
WHERE t.total_spend > a.avg_spend
ORDER BY t.total_spend DESC;

-- Q6: ranking window functions (rank customers inside their own city)
WITH customer_spend AS (
    SELECT c.customer_id,
           c.city,
           ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT customer_id,
       city,
       total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM customer_spend
ORDER BY city, row_num;

-- Q7: running total
-- Daily completed revenue plus a cumulative column
WITH daily AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day,
           ROUND(SUM(order_amount), 2) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day,
       revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily
ORDER BY day;
