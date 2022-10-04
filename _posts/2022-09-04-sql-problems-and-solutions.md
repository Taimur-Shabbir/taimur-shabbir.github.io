---
title: "SQL Ongoing Post: SQL Problems & Solutions to LeetCode and DataLemur Problems - 'Medium' & 'Hard' Difficulties"
date: 2022-09-04
tags: [SQL, PostgreSQL, MySQL]
#header: ""
classes: wide
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
header:
  image: /assets/img/brrr.png
  caption: ""
#image: blind.jpeg
excerpt: "A live document where I showcase my solutions to Medium and Hard-ranked SQL problems from LeetCode and DataLemur. "
#mathjax: "true"
---

**Note: To see the description for each problem, simply click on the problem name**

# DataLemur

## "Medium" Difficulty

### [First Transaction](https://datalemur.com/questions/sql-first-transaction)

``` sql

with t as(

select
      user_id,
      spend,
      dense_rank() over (partition by user_id order by transaction_date) as rank
from
      user_transactions
)

select count(distinct t.user_id) as users
from t
where t.rank = 1 and t.spend >= 50

```

### [Ad Campaign ROAS](https://datalemur.com/questions/ad-campaign-roas)

``` sql

select
    advertiser_id, round(((sum(revenue)/sum(spend))::DECIMAL), 2) as ROAS
from
    ad_campaigns
group by
      1

```

### [Average Review Ratings](https://datalemur.com/questions/sql-avg-review-ratings)

``` sql

select
      extract(month from submit_date) as month,
      product_id as product,
      round(avg(stars), 2) as avg_stars
from
      reviews
group by
      1, 2
order by
      1, 2

```

### [Top 5 Artists](https://datalemur.com/questions/top-fans-rank)

``` sql

select t.artist_name, t.rank as artist_rank

from(
select
      a.artist_name,
      dense_rank() over
      (order by sum(case when c.rank <= 10 then 1 else 0 end) DESC) as rank
from
    artists a
inner join
      songs b on a.artist_id = b.artist_id
inner join
      global_song_rank c on b.song_id = c.song_id
group by
      1) t

where t.rank <= 5
order by t.rank

```

### [Consulting Bench Time](https://datalemur.com/questions/consulting-bench-time)

```  sql
select
      s.employee_id,
      365 - sum(((c.end_date-c.start_date) + 1))
from
    staffing s
inner join
      consulting_engagements c on s.job_id = c.job_id
where
      s.is_consultant = 'true'
group by
      1
```


## "Hard" Difficulty

### [Y-on-Y Growth Rate](https://datalemur.com/questions/yoy-growth-rate)

``` sql

with table1 as(select
      product_id,
      extract(year from transaction_date) as year,
      sum(spend) as total_spend
FROM
      user_transactions
group by
      1, 2
ORDER BY
      1, 2)

SELECT
      a.year,
      a.product_id,
      a.total_spend as curr_year_spend,
      b.total_spend as prev_year_spend,
      round(((a.total_spend - b.total_spend)/(b.total_spend))*100.0, 2)
FROM
      table1 a
LEFT JOIN
      table1 b on a.product_id = b.product_id AND
                  a.year= b.year + 1

```


# LeetCode

## 'Medium' Difficulty

### [176. Second Highest Salary](https://leetcode.com/problems/second-highest-salary/)

``` sql

select
      ifnull(

      (select distinct
            salary
      from
            Employee
      order by
            Salary DESC
      limit
            1,1), null) as SecondHighestSalary

```

### [180. Consecutive Numbers](https://leetcode.com/problems/consecutive-numbers/)

```sql

select distinct
      a.num as ConsecutiveNums
from
      Logs a
join
      Logs b on a.id = b.id + 1
join
      Logs c on b.id = c.id + 1
where
      a.num = b.num and
      b.num = c.num
```

### [184. Department Highest Salary](https://leetcode.com/problems/department-highest-salary/)

``` sql
with sal as(

select
      d.id,
      d.name,
      max(e.salary) as maximum
from
      department d
inner join
      employee e on d.id = e.departmentId
group by
      1, 2)

select
      sal.name as department,
      e.name as employee,
      sal.maximum as salary
from
	    employee e
inner join
      sal on sal.id = e.departmentId and
      sal.maximum = e.salary

```

### [570. Managers with at Least 5 Direct Reports](https://leetcode.com/problems/managers-with-at-least-5-direct-reports/)

``` sql
select
      b.name as name
from
      Employee a
left join
      Employee b on a.managerId = b.id
where
      b.id is not null
group by
      b.name
having
      count(*) >=5

```

### [580. Count Student Number in Departments](https://leetcode.com/problems/count-student-number-in-departments/)

``` sql
select
      d.dept_name,
      count(s.student_name) as student_number
from
      Department d
left join
      student s on d.dept_id = s.dept_id
group by
      d.dept_name
order by
      student_number DESC, dept_name ASC

```

### [585. Investments in 2016](https://leetcode.com/problems/investments-in-2016/)

``` sql
select
      round(sum(tiv_2016), 2) as tiv_2016
from
      Insurance
where
      tiv_2015 not in


      (select
            tiv_2015
       from
            Insurance
       group by
            1
       having
            count(*) = 1)

and
      concat(lat, lon) not in

      (select
            concat(lat, lon)
       from
            Insurance
       group by
            lat, lon
       having
            count(*) > 1)

```

### [602. Friend Requests II: Who Has the Most Friends](https://leetcode.com/problems/friend-requests-ii-who-has-the-most-friends/)

``` sql
select distinct
      tab1.requester_id as id,
      sum(tab1.accepted) as num
from

(

      (select distinct
            requester_id,
            count(accepter_id) as accepted
       from
            RequestAccepted
       group by
            1)

       union all

      (select distinct
            accepter_id,
            count(requester_id) as requested
       from
            RequestAccepted
       group by
            1)

) tab1

group by
      tab1.requester_id
order by
      num DESC
limit
      1

```


### [608. Tree Node](https://leetcode.com/problems/tree-node/)

```sql

with cte1 as

(select
      a.p_id,
      a.id,
      b.p_id as b_p_id
from
      Tree a
left join
      Tree b on b.p_id = a.id)

select distinct
      cte1.id,
      case
          when cte1.p_id is null then "Root"
          when cte1.p_id is not null and cte1.b_p_id is not null then "Inner"
          else "Leaf"
          end as type
from
      cte1

```



### [612. Shortest Distance in a Plane](https://leetcode.com/problems/shortest-distance-in-a-plane/)

``` sql

with cte1 as(

select
      a.x as a_x,
      a.y as a_y,
      b.x as b_x,
      b.y as b_y
from
      Point2D a, Point2D b
where
      concat(a.x, ' ', a.y) != concat(b.x, ' ', b.y)
order by
      a_x, a_y)


select
      round(abs(min(sqrt(power((cte1.b_x-cte1.a_x), 2) + power((cte1.b_y-cte1.a_y), 2)))), 2) as shortest
from
      cte1
```

### [1045. Customers Who Bought All Products](https://leetcode.com/problems/customers-who-bought-all-products/)

```sql

with cte1 as(
select
      customer_id,
      count(distinct product_key) as total
from
	    Customer
group by
	    1)

select
      cte1.customer_id
from
      cte1
where
      cte1.total = (select count(distinct product_key) from Product)

```

### [1070. Product Sales Analysis III](https://leetcode.com/problems/product-sales-analysis-iii/)

```sql

select
      product_id,
      year as first_year,
      quantity,
      price
from
      Sales
where
      (product_id, year) in (select product_id, min(year) from Sales group by 1)

```


### [1077. Project Employees III](https://leetcode.com/problems/project-employees-iii/)

```sql

with cte1 as(

select
      p.project_id,
      max(e.experience_years) as max_exp
from
      Project p
inner join
      employee e using(employee_id)
group by
      p.project_id),

cte2 as(

select
      p.project_id,
      e.employee_id,
      e.name,
      e.experience_years
from
      Project p
inner join
      employee e using(employee_id))

select
      cte1.project_id,
      cte2.employee_id
from
      cte1, cte2
where
      cte1.project_id = cte2.project_id and
      cte1.max_exp = cte2.experience_years


```
