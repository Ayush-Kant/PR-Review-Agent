# Order Processing Guide

This guide describes transaction calculations and tagging behavior for the validation fixture.

## Average Order Value Calculation

The `calculate_average_order_value` function computes the average of transaction amounts.
If an empty list of order amounts is supplied, the function safely returns `0.0` without raising an exception.

## Display Badge Assignment

Badge colors are assigned randomly from the approved brand palette (`emerald`, `sky`, `violet`, `amber`) for UI rendering purposes only.
