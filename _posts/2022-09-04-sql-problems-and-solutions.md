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
  image: /assets/img/mkmk.jpeg
  caption: ""
#image: blind.jpeg
excerpt: "A live document where I demonstrate my solutions to Medium and Hard-ranked SQL problems from LeetCode and DataLemur. "
#mathjax: "true"
---

**Note: To see the description for each problem, simply click on the problem name**

# DataLemur

## "Medium" Difficulty

### Google - [Odd and Even Measurements](https://datalemur.com/questions/odd-even-measurements)


The key here is to use the ROW_NUMBER() window function and to use day only (using EXTRACT()) in the PARTITION BY clause while using the entire datetime value (named measurement_time) in the ORDER BY clause

```sql

with t as(

    SELECT
        measurement_id,
        measurement_value,
        measurement_time,
        row_number() over (partition by extract(day from measurement_time) order by measurement_time)
    FROM
        measurements)
SELECT
    measurement_time::date as measurement_day,
    sum(case WHEN row_number % 2 != 0 then measurement_value else 0 end) as odd_sum,
    sum(case when row_number % 2 = 0 then measurement_value else 0 end) as even_sum
FROM
    t
GROUP BY
    1
ORDER BY
    1

```


### UnitedHealth - [Patient Support Analysis (Part 4)](https://datalemur.com/questions/long-calls-growth)

The key to solving this problem is to use the LAG() window function to compare the number of long calls in one month to that of the previous month and to cast the percentage calculation as a decimal to be able to get negative percentages

```sql

with t as(

    SELECT
        extract(year from call_received) as year,
        extract(month from call_received) as month,
        sum(case when call_duration_secs > 300 then 1 else 0 end) as lc,
        lag( sum(case when call_duration_secs > 300 then 1 else 0 end), 1 )
        OVER( order by extract(year from call_received), extract(month from call_received) ) as previous_lc

    FROM
        callers
    GROUP BY
        1, 2
    ORDER BY
        1, 2)


SELECT
    year,
    month,
    round(((lc-previous_lc)/previous_lc::decimal*100), 1)
FROM
    t
```


### Spotify - [Spotify Streaming History](https://datalemur.com/questions/spotify-streaming-history)

The key here is to use UNION ALL instead of multiple CTEs

```sql

with t as(

    (SELECT
        user_id,
        song_id,
        count(song_id) as listens
    FROM
        songs_weekly
    WHERE
        extract(year from listen_time) <= 2022 and
        extract(month from listen_time) <= 8 and
        extract(day from listen_time) <= 4
    GROUP BY
        1, 2)

    union all


    (SELECT
        user_id,
        song_id,
        sum(song_plays) as listens
    FROM
        songs_history
    GROUP BY
        1, 2)

      )

SELECT
    user_id,
    song_id ,
    sum(listens) as song_plays
FROM
    t
GROUP BY
    1, 2
ORDER BY
    3 DESC

```


### Uber - [Second Ride Delay](https://datalemur.com/questions/2nd-ride-delay)

This question gave me more problems than usual and taught me about a crucial difference between rank() and row_number() which is obvious in hindsight. My approach here is as follows: find the order of trips per user_id dependent on the date. Isolate those user_id who booked a first ride on the same day as their registration ('in the moment' users). Find the difference between the 2nd ride date and the registration date for these latter users.

I kept getting the wrong answer initially until I finally realised I was using rank(), when I should have been using row_number().

If two values are equal (in this case, we are looking at ride dates), rank() will of course return the same value for both. However, row_number() will not; it will instead return the next value. For example, rank() will  return 1 and 1 for the same ride date but row_number() will return 1 and 2.

This realisation finally enabled me to get the solution


```sql

with ride_record as(

    SELECT
        u.user_id,
        u.registration_date,
        r.ride_date,
        row_number() over (partition by u.user_id order by r.ride_date) as trip_no
    FROM
        users u
    inner JOIN
        rides r on u.user_id = r.user_id),

    in_moment as(

    select DISTINCT
        ride_record.user_id
    FROM
        ride_record
    WHERE
        ride_record.registration_date = ride_record.ride_date)

SELECT
    ROUND(AVG(ride_date - registration_date),2) AS average_delay
FROM
    ride_record
inner JOIN
    in_moment on ride_record.user_id = in_moment.user_id
WHERE
    trip_no = 2

```

### CVS Health - [Pharmacy Analytics (Part 4)](https://datalemur.com/questions/top-drugs-sold)

```sql

with t as
    (SELECT
        manufacturer,
        drug,
        rank() over(partition by manufacturer order by sum(units_sold) DESC)
    FROM
        pharmacy_sales
    GROUP BY
        1, 2)
SELECT
    t.manufacturer,
    t.drug as top_drugs
FROM
    t
WHERE
    rank <= 2
ORDER BY
    t.manufacturer asc

```


### JPMorgan Chase - [Card Launch Success](https://datalemur.com/questions/card-launch-success)

```sql

with t as(

    SELECT
        card_name,
        issued_amount,
        rank() over (partition by card_name order by min(issue_year), min(issue_month))
    FROM
        monthly_cards_issued
    group by
        1, 2
    ORDER BY
        2 DESC)

SELECT
  card_name,
  issued_amount
FROM
    t
WHERE
    rank = 1

```

### Stitch Fix - [Repeat Purchases on Multiple Days](https://datalemur.com/questions/sql-repeat-purchases)

The following is not the most elegant solution but I had to think about this problem for longer than usual and I eventually got the answer.

The approach is to first find user_id and product_id combinations that occur over at least 2 different days. I accomplish this using COUNT(DISTINCT), GROUP BY and CONCAT. I use CONCAT because the purchase_date column has both the date and a timestamp, so simply using COUNT(DISTINCT) will not necessarily only capture different days as needed by the problem. It would also capture values for the same day, just at different times.

I use COUNT(DISTINCT) to then get only those user_id-product_id combinations which occur on at least 2 different days. Then it is a simple matter of wrapping these steps in a CTE and using COUNT(DISTINCT user_id) to find the number of users who made purchases on at least 2 different days.

``` sql
with t as(             -- begin CTE
    SELECT
        user_id,
        product_id,
        count(DISTINCT -- begin COUNT
            concat(    -- begin CONCAT
            extract(day from purchase_date),
            extract(month from purchase_date),
            extract(year from purchase_date)
                  )    -- end CONCAT
             )         -- end COUNT
    FROM
        purchases
    GROUP BY
        1, 2
    HAVING
        count(DISTINCT -- begin COUNT
            concat(    -- begin CONCAT
            extract(day from purchase_date),
            extract(month from purchase_date),
            extract(year from purchase_date)
                  )    -- end CONCAT
             ) > 1     -- end COUNT
         )             -- end CTE

SELECT
    count(distinct user_id) as repeat_purchasers
FROM
    t


```


### Etsy - [First Transaction](https://datalemur.com/questions/sql-first-transaction)

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

### Google - [Ad Campaign ROAS](https://datalemur.com/questions/ad-campaign-roas)

``` sql

select
    advertiser_id, round(((sum(revenue)/sum(spend))::DECIMAL), 2) as ROAS
from
    ad_campaigns
group by
      1

```

### Amazon - [Average Review Ratings](https://datalemur.com/questions/sql-avg-review-ratings)

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

### Spotify - [Top 5 Artists](https://datalemur.com/questions/top-fans-rank)

``` sql

with t as(

    SELECT
        artist_name,
        dense_rank() over (order by count(artist_name) DESC) as artist_rank
    FROM
        artists a
    inner JOIN
        songs s on a.artist_id = s.artist_id
    inner JOIN
        global_song_rank g on s.song_id = g.song_id
    WHERE
        g.rank <= 10
    GROUP BY
        1)

SELECT
    artist_name,
    artist_rank
FROM
    t
WHERE
    artist_rank <= 5

```

### Google - [Consulting Bench Time](https://datalemur.com/questions/consulting-bench-time)

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

### Facebook - [Active User Retention](https://datalemur.com/questions/user-retention)

The following shows my first solution which has more parts to it than necessary. After it we can see my second solution, which pares down the unnecessary parts of the query and makes it cleaer.

The approach here is as follows. I need to use a self-join on the user_actions table so that I can compare, for the same user_id, whether or not the user_id had transactions in consecutive months. The question defines "active users" as those who have a required interaction type in both July and June, so I want to look for these users only.

I will join using a compound key where a.user_id is equal to b.user_id and also the event_date of table alias b occurs after the event date of table alias b.

Next, I need to restrict this self join to only those cases where the difference between b.event_date and a.event_date is only 1 month, again because we are interested in July 2022 and June 2022 solely, and where the interaction type in both months is one of "sign-in", "like", or "comment".

After this is done, I wrap this query in a CTE. From this CTE, I find the COUNT of DISTINCT user_id values of those users whose second interaction occurred in July 2022; by necessity, this means their first interaction occurred in June 2022 due to the way our CTE is set up. This is exactly what we're looking for.

``` sql

-- First solutions

with t as(

    SELECT
          a.user_id,
          a.event_id as event_id_a,
          a.event_date as a_date,
          b.event_id as event_id_b,
          b.event_date as b_date
    FROM
          user_actions a
    JOIN
          user_actions b on a.user_id = b.user_id AND b.event_date > a.event_date
    WHERE
          (DATE_PART('year', b.event_date::timestamp) - DATE_PART('year', a.event_date::timestamp)) * 12 +
          (DATE_PART('month', b.event_date::timestamp) - DATE_PART('month', a.event_date::timestamp)) = 1 AND
          a.event_type in ('sign-in', 'like', 'comment') AND
          b.event_type in ('sign-in', 'like', 'comment')
         )
SELECT
    extract(month from t.b_date) as month,
    count(distinct t.user_id) as monthly_active_users
FROM
    t
WHERE
     extract(month from t.b_date) = 7 AND
     extract(year from t.b_date) = 2022
GROUP BY
    1

```

A less wordy, cleaner solution

``` sql


    SELECT
          extract(month from b.event_date) as month,
          count(distinct a.user_id) as monthly_active_users
    FROM
          user_actions a
    JOIN
          user_actions b on a.user_id = b.user_id AND b.event_date > a.event_date
    WHERE

          (DATE_PART('month', b.event_date::timestamp) - DATE_PART('month', a.event_date::timestamp)) = 1 AND
          a.event_type in ('sign-in', 'like', 'comment') AND
          b.event_type in ('sign-in', 'like', 'comment') AND
          extract(month from b.event_date) = 7 AND
          extract(year from b.event_date) = 2022
    GROUP BY
        1

```

### Wayfair - [Y-on-Y Growth Rate](https://datalemur.com/questions/yoy-growth-rate)

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


### [1264. Page Recommendations](https://leetcode.com/problems/page-recommendations/description/)

The key to solving this problem is to use a CASE statement to collect all the user_id of individuals who are friends of user_id = 1 in a CTE. All that is left to do is to join the Likes table on this CTE to find out what pages the friends of user_id = 1 like and filter out those pages that user_id = 1 already Likes

```sql

with rel as(

    select
        1 as user1,
        case when user1_id = 1 then user2_id
             when user2_id = 1 then user1_id end as user1_friends
    from
        Friendship
)

select distinct
    l.page_id as recommended_page
from
    rel r
inner join
    likes l on r.user1_friends = l.user_id
where
    l.page_id not in (select page_id from Likes where user_id = 1)

```

### [1205. Monthly Transactions II](https://leetcode.com/problems/monthly-transactions-ii/description/)

This is one of the tricker Medium-difficulty questions I've come across. The trickiness lay in the fact that we need to recognise that chargebacks for a given transaction may not occur in the same month in which the transaction itself occured.

This means I needed to use UNION to collect all transactions (those that were approved, decline and charged back) and their months in a CTE, which I could then query to get the required result.

Finally, the HAVING clause exists comply with the condition of the question which asks us to ignore rows where all of the latter 4 columns are 0


``` sql

with t1 as       
        (select
            id,
            date_format(trans_date, '%Y-%m') as month,
            country,
            state as status,
            amount
        from
            transactions

        union

        select
           c.trans_id,
           date_format(c.trans_date, '%Y-%m') as month,
           t.country,
           'chargeback' as status,
           t.amount
        from
            chargebacks c
        inner join
            transactions t on c.trans_id = t.id
        )

select
    t1.month,
    t1.country,
    ifnull(sum(case when status = 'approved' then 1 else 0 end), 0) as approved_count,
    ifnull(sum(case when status = 'approved' then amount else 0 end), 0) as approved_amount,
    ifnull(sum(case when status = 'chargeback' then 1 else 0 end), 0) as chargeback_count,
    ifnull(sum(case when status = 'chargeback' then amount else 0 end), 0) as chargeback_amount
from
    t1
group by
    1, 2
having
    approved_count > 0 or
    approved_amount > 0 or
    chargeback_count > 0 or
    chargeback_amount > 0

```

### [1107. New Users Daily Count](https://leetcode.com/problems/new-users-daily-count/description/)

The key here was to use DATEDIFF in the HAVING clause of the subquery with min(activity_date) instead of in the WHERE clause of the subquery (which would give the wrong answer because we cannot use min() in the WHERE clause)

``` sql

select
    t.login_date,
    count(distinct t.user_id) as user_count
from

(select
    user_id,
    min(activity_date) as login_date
from
    Traffic
where
    activity = 'login'
group by
    1
having
    abs(datediff('2019-06-30', min(activity_date))) <= 90 ) t

group by
    1


```


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
