---
title: "Ongoing Post: SQL Problems & Solutions to LeetCode and DataLemur Problems - 'Medium' & 'Hard' Difficulties"
date: 2022-09-04
tags: [SQL, PostgreSQL, MySQL]
#header: ""
classes: wide
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
header:
  image: /assets/img/blind.jpeg
  caption: ""
#image: blind.jpeg
excerpt: "A live document where I showcase my solutions to Medium and Hard-ranked SQL problems from LeetCode and DataLemur. "
#mathjax: "true"
---

# DataLemur

## "Medium" Difficulty

###[First Transaction](https://datalemur.com/questions/sql-first-transaction)

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

``` SQL

select
    advertiser_id, round(((sum(revenue)/sum(spend))::DECIMAL), 2) as ROAS
from
    ad_campaigns
group by
      1

```

### [Average Review Ratings](https://datalemur.com/questions/sql-avg-review-ratings)

```SQL

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

``` SQL

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

```sql
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

``` SQL

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
