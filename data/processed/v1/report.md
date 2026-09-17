# Dataset report

- Rows read: **2484**
- Records kept: **2468**
- Quarantined: **16**
- Classes: **24** (taxonomy v1)
- Duplicate clusters collapsed: **14**
- Reconciliation (`read == kept + quarantined`): **OK**
- Split seed: `42`

## Splits

| Split | Records |
| --- | ---: |
| train | 1727 |
| validation | 370 |
| test | 371 |

## Class distribution

Smallest class: 22 · largest: 119 · imbalance ratio: 5.41x

| Class | Total | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| Accountant | 118 | 83 | 17 | 18 |
| Advocate / Legal | 117 | 82 | 17 | 18 |
| Agriculture | 62 | 43 | 10 | 9 |
| Apparel | 97 | 68 | 15 | 14 |
| Arts | 103 | 72 | 16 | 15 |
| Automobile | 36 | 25 | 6 | 5 |
| Aviation | 116 | 81 | 17 | 18 |
| Banking | 115 | 81 | 17 | 17 |
| Business Development | 118 | 83 | 17 | 18 |
| Business Process Outsourcing | 22 | 15 | 4 | 3 |
| Chef / Culinary | 116 | 81 | 17 | 18 |
| Construction | 112 | 78 | 17 | 17 |
| Consultant | 114 | 80 | 17 | 17 |
| Designer | 107 | 75 | 16 | 16 |
| Digital Media | 96 | 67 | 15 | 14 |
| Engineering | 116 | 81 | 17 | 18 |
| Finance | 116 | 81 | 17 | 18 |
| Fitness | 117 | 82 | 17 | 18 |
| Healthcare | 114 | 80 | 17 | 17 |
| Human Resources | 109 | 76 | 16 | 17 |
| Information Technology | 119 | 83 | 18 | 18 |
| Public Relations | 111 | 78 | 17 | 16 |
| Sales | 115 | 81 | 17 | 17 |
| Teacher / Education | 102 | 71 | 16 | 15 |

## Text length

| Metric | Characters | Words |
| --- | ---: | ---: |
| min | 688 | 107 |
| p05 | 2,183 | 296 |
| p25 | 4,814 | 645 |
| median | 5,538 | 751 |
| p75 | 6,839 | 925 |
| p95 | 10,296 | 1,396 |
| max | 35,922 | 5,146 |
| mean | 5,929 | 806 |

## Quarantine reasons

| Reason | Count |
| --- | ---: |
| `EXACT_DUPLICATE` | 2 |
| `NEAR_DUPLICATE` | 13 |
| `TEXT_EMPTY` | 1 |
