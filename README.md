# Multi-Period Disaster Logistics Planning

This project implements a A Stochastic Service Network Design Model for Disaster Logistics Planning. The model integrates:

* Two-stage stochastic programming (2SSP)
* Benders decomposition
* Rolling horizon framework
* MVP (mean value problem) approximation

---

## Folder Structure

```
multi_period_disaster_logistics/
├── main.py                        <- Entry point to run the model
├── demand_data.py                <- Scenario-based demand processing
├── network_data.py               <- Loads network info (nodes, arcs, distances)
├── model.py                      <- Extended formualtion, rolling horizon framework
├── method.py                     <- Benders decomposition
├── output.py                     <- Result formatting and Excel export
├── fixed_params.yaml             <- Model parameters
├── Data/                         <- Input files (distances, populations, etc.)
├── Generated data/              <- Auto-created simulation demand files
├── Results/                      <- Output Excel results
├── MSSP_IHRLEP_Networks/        <- External methods required for demand estimation
```

---

## Setup & Requirements

1. **Python Version**: Python 3.10+
2. **Dependencies**:

   ```bash
   pip install pandas numpy gurobipy shapely openpyxl geopandas
   ```
3. **Solver**:

   * Gurobi installed with academic license

---

## How to Run

The main entry point is `main.py`, which supports 4 solution methods:

| Method   | Flag    | Description                |
| -------- | ------- | -------------------------- |
| Extended | `-sm 1` | Full scenario-based 2SSP   |
| Benders  | `-sm 2` | Decomposition approach     |
| MVP      | `-sm 3` | Mean value problem         |
| RH       | `-sm 4` | Rolling horizon evaluation |

Example:

```bash
python main.py -ns 2 -iv original -sm 1
```

**Arguments**:

* `-ns` : Number of in-sample demand scenarios
* `-iv` : Instance version (e.g., `original`)
* `-sm` : Solution method (1–4)
* `-tl` : Time limit (for Benders)
* `-eos` : If set, evaluates OOS scenarios for S-2SSP
* `-oos` : Size of out-of-sample scenarios for S-2SSP evaluation
* `-no` : Size of out-of-sample scenarios for RH-2SSP evaluation

---

## Output

Results are written to the `Results/` folder in Excel format:

* `ext_results.xlsx`
* `benders_results.xlsx`
* `mvp_results.xlsx`
* `rh_results.xlsx`
* `oos_2SSP_results.xlsx`

