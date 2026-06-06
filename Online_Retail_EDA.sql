create database ecommerce_db;

#ONLINE RETAIL E-COMMERCE EDA


-- Unique countries
SELECT COUNT(DISTINCT Country) AS Unique_Countries
FROM customer_transactions;

-- Unique products
SELECT COUNT(DISTINCT Description) AS Unique_Products
FROM customer_transactions;

-- Total customers
SELECT COUNT(DISTINCT CustomerID) AS Total_Unique_Customers
FROM customer_transactions;

-- Total transactions
SELECT COUNT(DISTINCT InvoiceNo) AS Total_Transactions
FROM customer_transactions;

-- Total revenue
SELECT ROUND(SUM(TotalSales), 2) AS Gross_Revenue
FROM customer_transactions;

-- Average Order Value (AOV)
SELECT ROUND(
       SUM(TotalSales) /
       COUNT(DISTINCT InvoiceNo), 2
       ) AS Average_Order_Value
FROM customer_transactions;


-- Top 10 customers by revenue
SELECT CustomerID,
       Country,
       ROUND(SUM(TotalSales), 2) AS Total_Spent
FROM customer_transactions
WHERE CustomerID <> -1
GROUP BY CustomerID, Country
ORDER BY Total_Spent DESC
LIMIT 10;

-- Revenue contribution of Top 10 customers
SELECT
ROUND(
    SUM(CustomerRevenue) * 100 /
    (SELECT SUM(TotalSales)
     FROM customer_transactions),
    2
) AS Top_10_Revenue_Share_Percentage
FROM (
    SELECT SUM(TotalSales) AS CustomerRevenue
    FROM customer_transactions
    WHERE CustomerID <> -1
    GROUP BY CustomerID
    ORDER BY CustomerRevenue DESC
    LIMIT 10
) t;


-- Repeat vs One-Time Customers
SELECT
CASE
WHEN OrderCount = 1
THEN 'One-Time Customer'
ELSE 'Repeat Customer'
END AS CustomerType,
COUNT(*) AS CustomerCount,
ROUND(
COUNT(*) * 100 /
(
SELECT COUNT(DISTINCT CustomerID)
FROM customer_transactions
WHERE CustomerID <> -1
),2
) AS Customer_Percentage
FROM
(
SELECT CustomerID,
       COUNT(DISTINCT InvoiceNo) AS OrderCount
FROM customer_transactions
WHERE CustomerID <> -1
GROUP BY CustomerID
) t
GROUP BY CustomerType;


-- Top products by quantity sold
SELECT Description,
       SUM(Quantity) AS Total_Units_Sold
FROM customer_transactions
GROUP BY Description
ORDER BY Total_Units_Sold DESC
LIMIT 10;

-- Top products by revenue
SELECT StockCode,
       UPPER(Description) AS Product_Description,
       ROUND(SUM(TotalSales), 2) AS Generated_Revenue
FROM customer_transactions
GROUP BY StockCode, Description
ORDER BY Generated_Revenue DESC
LIMIT 20;


-- Monthly revenue and orders
SELECT DATE_FORMAT(InvoiceDate, '%Y-%m') AS Month,
       COUNT(DISTINCT InvoiceNo) AS Monthly_Orders,
       ROUND(SUM(TotalSales), 2) AS Monthly_Revenue
FROM customer_transactions
GROUP BY Month
ORDER BY Month;

-- Revenue by month
SELECT DATE_FORMAT(InvoiceDate, '%Y-%m') AS Month,
       ROUND(SUM(TotalSales), 2) AS Revenue
FROM customer_transactions 
group BY Month
ORDER BY Revenue DESC;

-- Orders by month
SELECT DATE_FORMAT(InvoiceDate, '%Y-%m') AS Month,
       COUNT(DISTINCT InvoiceNo) AS Order_Volume
FROM customer_transactions
GROUP BY Month
ORDER BY Order_Volume DESC;


-- Revenue and customers by country
SELECT Country,
       COUNT(DISTINCT CustomerID) AS Active_Customer_Count,
       COUNT(DISTINCT InvoiceNo) AS Total_Orders,
       ROUND(SUM(TotalSales), 2) AS Total_Revenue,
       ROUND(
       (SUM(TotalSales) /
       (SELECT SUM(TotalSales)
        FROM customer_transactions)) * 100,
       2
       ) AS Global_Revenue_Share_Percentage
FROM customer_transactions
GROUP BY Country
ORDER BY Total_Revenue DESC;

-- Average Order Value by Country
SELECT Country,
       COUNT(DISTINCT InvoiceNo) AS Order_Count,
       ROUND(
       SUM(TotalSales) /
       COUNT(DISTINCT InvoiceNo),
       2
       ) AS Country_Average_Order_Value
FROM customer_transactions
GROUP BY Country
ORDER BY Country_Average_Order_Value DESC;

