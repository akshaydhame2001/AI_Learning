-- 25 Most important SQL questions

-- 1. Write a query to find the second highest salary in an employee table.
SELECT MAX(salary) As SecondHighestSalary
FROM employee
WHERE salary < (SELECT MAX(salary) FROM employee);
-- OR in MYSQL
SELECT DISTINCT salary FROM employee
ORDER BY salary DESC
LIMIT 1 OFFSET 1;

-- 2. Fetch all employees whose names contain the letter "a" exactly twice.
SELECT *
FROM employee
WHERE LENGTH(name) - LENGTH(REPLACE(LOWER(name), 'a', '')) = 2;
-- CHAR_LENGTH in MYSQL

-- 3. How do you retrieve only duplicate records from a table.
SELECT column1, column2
FROM employee
GROUP BY column1, column2
HAVING COUNT(*) > 1;

--- 4. Write a query to calculate the running total of sales by date.
SELECT 
  sale_date,
  sales_amount,
  SUM(sales_amount) OVER (ORDER BY sale_date) AS running_total
FROM sales;

-- 5. Find the employee who earn more than the average salary in their department.
SELECT *
FROM employee e
WHERE e.salary > (
  SELECT AVG(salary)
  FROM employee
  WHERE department = e.department
);

-- 6. Write a query to find the most frequently occurring value in a column.
SELECT column1, COUNT(*) AS frequency
FROM employee
GROUP BY column1
ORDER BY frequency DESC
LIMIT 1;

-- 7. Fetch records where the date is within the last 7 days from today.
SELECT *
FROM employee
WHERE employee_date >= DATE('now', '-7 days');
-- MySQL: CURDATE() - INTERVAL 7 DAY, PostgreSQL: CURRENT_DATE - INTERVAL '7 days'

-- 8. Wriet a query to count how many employees share the same salary.
SELECT salary, COUNT(*) AS employee_count
FROM employee
GROUP BY salary
HAVING COUNT(*) > 1
ORDER BY employee_count DESC;

-- 9. How do you fetch the top 3 records for each group in a table?
SELECT *
FROM (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY department ORDER BY revenue DESC) AS rn
  FROM sales
) AS ranked
WHERE rn <= 3;

-- 10. Retrieve products that were never sold(hint: use LEFT JOIN)
SELECT p.*
FROM products p
LEFT JOIN sales s ON p.product_id = s.product_id
WHERE s.product_id IS NULL;
