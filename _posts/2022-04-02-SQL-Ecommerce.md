---
title: "SQL & Tableau Project: - Analysing Performance of an E-Commerce Business "
date: 2022-04-02
tags: [SQL, Database Management, Tableau, Data Visualisation]
classes: wide
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
excerpt: "SQL queries, dashboards and toystores"
header:
  image: /assets/img/mike-petrucci-c9FQyqIECds-unsplash.jpg
  caption: "Photo by [Mike Petrucci](https://unsplash.com/@mikepetrucci?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/ecommerce?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)"
#mathjax: "true"
---



## Note: This project displays the following SQL skills:

- Filtering data
- Joins
- Grouping and Aggregation
- CTEs
- Derived tables
- Subqueries
- Views



# Introduction

This analysis is conducted on a Mexican E-commerce store called Maven Toys and uses SQL and Tableau for querying the data and visualisation, respectively.

The purpose of this is to extract insights that can shed light on current business performance and better inform future business decisions. These may range from learning about things such as which store to promote, which products tend to perform better in some months compared to others, whether or not money tied up in inventory could be managed better and more.

The queries are in 3 parts:

- Products and Profitability
- Seasonality
- City-level and Stock Analysis

A short summary of this project, including information on the data tables, columns, data types and data source can be found in a GitHub repository [README](https://github.com/Taimur-Shabbir/SQL-Ecommerce-Analysis) file.

There is also a [dashboard](https://public.tableau.com/app/profile/taimur.shabbir/viz/E-CommerceInProgress/Dashboard1) visualising some of the queries that are present below. But all of the visualisations in the dashboard can also be found below, attached to the relevant query.

# Part 1 - Data Cleaning

Here I'm going to remove the dollar sign '$' in the 'product_cost' and 'product_price' columns, then change type to 'double'. This is done so that we can perform calculations with the data in these columns.

~~~~SQL

update products
set product_cost = replace(product_cost, '$','')

update products
set product_price = replace(product_price, '$','')

alter table products
modify product_cost double

alter table products
modify product_price double

~~~~

# Part 2 - Products and Profitability

##  Q1) Which 5 products have generated the most profit?

```SQL

select
      a.product_name,
      round(sum((a.product_price-a.product_cost)*b.units), 2) as total_profit
from
      products a
inner join
      sales_toys b using(Product_ID)
group by
      a.product_name
order by
      total_profit desc
limit
      5
```
<img src="{{ site.url }}{{ site.baseurl }}/images/ecommerceImages/most_profitable_products.png" alt="None">

Colorbuds, which are a type of earphones, were more profitable than the next two most profitable products combined.


### Interpretation & Takeaway

Colorbuds account for 43% of all profit, which is very significant. If combined with Action Figures, the second highest
profitable product, nearly 62% of all profit is accounted for.

It may be tempting to recommend that Maven Toys should consider focusing on these two products in marketing campaigns,
for example, at the expense of other products. However, without further information, we cannot make this recommendation.

For example, it may be the case that one or more of the other 3 products serves as a loss leader. Loss leaders, as we
know, are products that are intentionally sold at a lower price point, sometimes even below cost price, in order to attract
customers to the company. The company can sell cross-sell other, higher-priced/more profitable products to the same
customers as a result. 

Therefore, the presence of the loss leader may actually be critical to overall profits because without them, the company
may not attract as many customers in the first place. It may be the case that one of Lego Bricks, Deck of Cards & Glass 
Marbles serves as a loss leader. Completely neglecting one of these in favour of the 2 most profitable products may result
in lower overall profits.

In conclusion, if the company has knowingly placed one of these products as a loss leader then ceasing its promotion is
not advisable. Conversely, if this is not the case, then the company may run an A/B test online with an ad campaign to
verify whether further promoting Colorbuds or Action Figures leads to an increase in overall profit

## Q2) What product category is the most profitable overall?

~~~SQL

select
      a.product_category,
      round(sum((a.product_price-a.product_cost)*b.units), 2) as total_profit
from
      products a
inner join
      sales_toys b using(Product_ID)
group by
      a.product_category
order by
      total_profit desc
~~~


## Q3) What products are high margin and which ones are low margin?


```SQL
select
    product_name, round(product_price-product_cost, 2) as margin,
    case when (product_price-product_cost) > 5 then 'High Margin' else 'Low Margin' end as Status
from
    products
order by
    margin desc
```

## Q4) Are high margin products more profitable? Or are low margin products more profitable?


```SQL

-- find the total profit and margin for each product

with cte1 as(

    select
          a.product_name,
          round(sum((a.product_price-a.product_cost)*b.units), 2) as total_profit,
          max(round(a.product_price-a.product_cost, 2)) as margin
    from
          products a
    inner join
          sales_toys b using(Product_ID)
    group by
          a.product_name
    order by
          total_profit desc)

select
    cte1.product_name,
    cte1.total_profit,
    cte1.margin, # get the results from above CTE and label each product as 'high' or 'low' margin
    case when cte1.margin > 5 then 'High Margin' else 'Low Margin' end as Status
from
    cte1
order by
    total_profit DESC, margin desc
```

<img src="{{ site.url }}{{ site.baseurl }}/images/ecommerceImages/margins_vs_profit.png" alt="None">


### Interpretation & Takeaway

We see that top 2 most profitable products, Colorbuds and Action Figures, are considered to be high-margin products
while the next 3 most profitable products are low margin.

The criteria of a high margin product is that the selling price is at least $6 greater than the cost price. I chose this
figure based on the distribution of margins across all products, but it is still fairly arbitrary.

The graph shows us that beyond these first 2 products, there seems to be little to no correlation between profit and 
whether or not the product is low- or high-margin. It may be the case that there is an addition factor, beyond the fact
that they are high-margin products, that may be influencing why Colorbuds and Action Figures are so profitable.

For now, I do not recommend changing the price points for any of the products or differentiating between high- and
low-margin products when it comes to promotion.

# Part 3 - Seasonality

##  Q5) Is there a seasonality aspect to sales?

The date range of the data is 2017-01-01 to 2018-09-30. This means that all months apart from October, November and December have two months worth of data (once for 2017 and once for 2018).


```SQL

select DATE_FORMAT(a.date, '%Y-%m'),
     round(sum(a.Units*b.product_price), 2) as total_monthly_revenue
from
     sales a 
inner join 
     products b on a.Product_ID = b.Product_ID
group by 
     DATE_FORMAT(a.date, '%Y-%m')


```

<img src="{{ site.url }}{{ site.baseurl }}/images/ecommerceImages/revised_seasonality.png" alt="None">


### Interpretation & Takeaway

As we can see, there is a general uptick in sales revenue for Maven Toys during the Spring months. Apart from a sharp rise
in revenue in December, which can be attributed to Christmas and New Year's holidays, revenue is generally lower in the 
Winter and Autumn months than in Spring. 

One possible recommendation here for Maven Toys is to introduce new products that are likely to sell better in Winter &
Autumn. For example, toy versions of skis for children or snowman toys could prove to be popular during these times.

Overall, these observations do seem to indicate there is an element of seasonality involved. However, we still need to
be careful in making generalisations because we have less than 2 year of data. Our conclusions would be more valid if we
had, say, 5 years' worth of data. But for now it seems that the spring months are correlated with an uptick 
in sales for Maven Toys


## Q6) Is there a difference in product category popularity between Summer and Winter months?

```SQL

-- get revenue of winter months

with w as(

    select
          a.product_category,
          round(sum(a.product_price*b.units), 2) as total_revenue
    from
          products a inner join sales_toys b using(Product_ID)
    where
          month(b.date) in(1, 2, 12)
    group by
          a.product_category),

-- get revenue of summer months

s as(

    select
          a.product_category,
          round(sum(a.product_price*b.units), 2) as total_revenue
    from
          products a inner join sales_toys b using(Product_ID)
    where
          month(b.date) in(6, 7, 8)
    group by
          a.product_category)

-- collect the results of the above two CTEs, joining on product category

select
    w.product_category,
    w.total_revenue as winter_revenue,
    s.total_revenue as summer_revenue
from
    w
inner join
    s using(product_category)
```

<img src="{{ site.url }}{{ site.baseurl }}/images/ecommerceImages/winter_v_summer.png" alt="None">

### Interpretation & Takeaway

Here we see some interesting phenomena.

All categories bar Games sell better in Summer than in Winter. This makes sense as people are less likely to pursue
outdoor activities in the Winter than in Summer, and games are primarily an indoor-oriented hobby/pastime. 

Toys and Sports & Outdoors have the biggest differences in sales between Summer and Winter. Again, this is to be expected
given most people's inclination towards outdoor activities when the whether is warmer. In Summer, Toys sales are
24% greater than in Winter. This figure is about 54% for Sports & Outdoors

Electronics sales are nearly the same in Winter as in Summer, with Summer sales slightly higher

Finally, Games sales are nearly 12% higher in Winter than in Summer

# Part 4 - City-level and Stock analysis

## Q7) What cities generate the most revenue?

```SQL
select
      s.Store_city,
      round(sum(p.product_price*t.units), 2) as City_Revenue
from
      stores s
inner join
      sales_toys t on s.Store_ID = t.Store_ID
inner join
      products p on p.Product_ID = t.Product_ID
group by
      s.Store_City
order by
      City_Revenue desc
```

<img src="{{ site.url }}{{ site.baseurl }}/images/ecommerceImages/revenue_by_city.png" alt="None">

### Interpretation & Takeaway

Maven Toys have their highest sales on a per-city basis in Mexico City (Ciudad de Mexico), Guadalajara and Monterrey.
After cross-referencing the list of cities in which Maven Toys has branches with a list of the largest cities in Mexico
by population, we can see that the largest cities in population terms are associated with the greatest sales.

Again, this is not a surprising finding as a large population means more potential customers. Larger cities also tend to
be more urban and richer on average, so this may also have an influence.

What's more interesting is that all the other cities in which Maven Toys has a presence are not counted among the 30 
largest cities by population. What this could imply is that Maven Toys may consider opening a branch in a city with among
the largest populations in Mexico, such as Tijuana or Leon, since population size and sales seem to be correlated for
them. 



## Q8) Are older (legacy) stores able to incorporate ecommerce technology well?

What we are asking is: are legacy stores providing the same level of success as newer stores? The earliest store opening is in 1992 and the latest is in 2016. Let's consider 2004 which is halfway between the two dates as the cutoff point.

```SQL

with cte2 as(

      select
            s.Store_ID,
            s.Store_Name,
            round(sum(p.product_price*t.units), 2) as Store_Revenue
      from
            stores s
      inner join
            sales_toys t on s.Store_ID = t.Store_ID
      inner join
            products p on p.Product_ID = t.Product_ID
      group by
            s.Store_ID, s.Store_Name)

select
      cte2.Store_name,
      cte2.Store_Revenue,
      case when year(s.store_open_date) <= 2004 then 'Legacy Store' else 'New Store' end as Age_Status
from
      cte2
inner join
      stores s using(Store_ID)
order by
      Store_Revenue DESC

```

## Q9) How much money is tied up in each store in the form of stock? Top 5 vs Bottom 5

Top 5:

```SQL

select
      i.store_id,
      s.store_name,
      round(sum(i.Stock_On_Hand*p.product_price), 2) as total_inventory_value
from
      inventory i
inner join
      stores s on i.store_id = s.store_id
inner join
      products p on i.product_id = p.product_id
group by
      i.store_id, s.store_name
order by
      total_inventory_value desc
limit 5

```

Bottom 5:

```SQL

select
      i.store_id,
      s.store_name,
      round(sum(i.Stock_On_Hand*p.product_price), 2) as total_inventory_value
from
      inventory i
inner join
      stores s on i.store_id = s.store_id
inner join
      products p on i.product_id = p.product_id
group by
      i.store_id, s.store_name
order by
      total_inventory_value asc
limit 5
```

## Q10) Is there any relationship between value of stock tied up and revenue by store?

```SQL

-- get inventory value per store

with inventory_value as(

      select
            i.store_id,
            s.store_name,
            round(sum(i.Stock_On_Hand*p.product_price), 2) as total_inventory_value
      from
            inventory i
      inner join
            stores s on i.store_id = s.store_id
      inner join
            products p on i.product_id = p.product_id
      group by
            i.store_id, s.store_name
      order by
            total_inventory_value),[2022-03-27-SQL-Coronavirus.md](2022-03-27-SQL-Coronavirus.md)

-- get total revenue per store

store_revenue as(

      select
            s.store_name,
            sum(p.product_price*t.units) as revenue
      from
            products p
      inner join
            sales_toys t on p.product_id = t.product_id
      inner join
            stores s on t.store_id = s.store_id
      group by
            s.store_name)

-- collect and return relevant results of above CTEs, joining on store_name

select
      inventory_value.store_name,
      inventory_value.total_inventory_value,
      store_revenue.revenue
from
      inventory_value
inner join
      store_revenue using(store_name)
order by
      store_revenue.revenue desc
```

<img src="{{ site.url }}{{ site.baseurl }}/images/ecommerceImages/Inventory_Value_vs_Sales.png" alt="None">


### Interpretation & Takeaway

There seems to be no clear correlation between the value of inventory in stock and revenue. Normally we can expect to see
lower revenue when value of inventory is stock is high. Moreover, high levels of stock are seen as a business cost and are 
generally undesirable for a business. However, that correlation does not seem to hold here.

Value of inventory in stock for most stores ranges between $6,000 and $7,000, but Revenues for those same stores range
between $200,000 and $440,000.