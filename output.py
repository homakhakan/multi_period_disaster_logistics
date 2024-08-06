import pandas as pd

class OptimizationModel(OptimizationModel):  # Assuming it extends OptimizationModel from model.py
    def get_results(self):
        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.TIME_LIMIT:
            # Collect general results
            results_data = {
                'Objective Value': [self.model.objVal],
                'Penalty Cost': [round(sum(sum(self.model_setup.params['p_i'][i] * self.U[i, t].x for i in range(self.model_setup.V)) for t in range(self.model_setup.T)), 4) * self.model_setup.tperiod],
                'Total Cost': [round(self.b.x, 2)],
                'Final Gap Value': [self.model.MIPGap],
                'Weight of Cost': [self.model_setup.weight_cost],
                'Weight of Penalty': [self.model_setup.weight_penalty]
            }

            results_df = pd.DataFrame(results_data)

            # Collect results for shelters
            shelter_data = []
            for i in self.model_setup.V1:
                shelter_data.append([
                    i,
                    round(sum(self.S[i, t].x for t in range(self.model_setup.T)), 2),
                    round(self.U[i, self.model_setup.T - 1].x, 2),
                    round(sum(self.model_setup.params['d_shelter'][i - self.model_setup.V1[0]][t] for t in range(self.model_setup.T)), 2)
                ])

            shelter_df = pd.DataFrame(shelter_data, columns=['Location', 'Total Satisfied', 'Total Shortage', 'Total Demand'])

            # Collect results for PoDs
            pod_data = []
            for i in self.model_setup.V3:
                pod_data.append([
                    i,
                    round(sum(self.S[i, t].x for t in range(self.model_setup.T)), 2),
                    round(self.U[i, self.model_setup.T - 1].x, 2),
                    round(sum(self.model_setup.params['d_pod'][i - self.model_setup.V3[0]][t] for t in range(self.model_setup.T)), 2)
                ])

            pod_df = pd.DataFrame(pod_data, columns=['Location', 'Total Satisfied', 'Total Shortage', 'Total Demand'])

            # Collect detailed results for a specific shelter
            i = 22  # Shelter check
            shelter_check_data = []
            for t in range(self.model_setup.T):
                shelter_check_data.append([
                    t,
                    round(self.S[i, t].x, 2),
                    round(self.U[i, t].x, 2),
                    round(self.model_setup.params['d_shelter'][i - self.model_setup.V1[0]][t], 2),
                    round(self.I[i, 2, t].x, 2),
                    round(self.I[i, 3, t].x, 2),
                    round(self.I[i, 4, t].x, 2),
                    round(self.I[i, 5, t].x, 2),
                    round(sum(self.f[j, i, 2, t].x for j in range(self.model_setup.V) if (j, i) in self.model_setup.A), 2),
                    round(sum(self.f[j, i, 3, t].x for j in range(self.model_setup.V) if (j, i) in self.model_setup.A), 2),
                    round(sum(self.f[j, i, 4, t].x for j in range(self.model_setup.V) if (j, i) in self.model_setup.A), 2),
                    round(sum(self.f[j, i, 5, t].x for j in range(self.model_setup.V) if (j, i) in self.model_setup.A), 2),
                    round(sum(sum(self.K[j, i, r, t].x for j in range(self.model_setup.V) if (j, i) in self.model_setup.A) for r in range(self.model_setup.R)), 2),
                    round(sum(sum(self.K[i, j, r, t].x for j in range(self.model_setup.V) if (i, j) in self.model_setup.A) for r in range(self.model_setup.R)), 2),
                    round(self.J[i, t].x, 2)
                ])

            shelter_check_df = pd.DataFrame(shelter_check_data, columns=[
                'Time', 'Satisfied', 'Shortage', 'Demand', 'FoodInv', 'WaterInv', 'CotInv', 'BlanketInv', 'FoodRec', 'WaterRec', 'CotRec', 'BlanketRec', 'TruckIn', 'TruckOut', 'TruckInv'
            ])

            # Collect detailed results for a specific PoD
            i = 36  # PoD check
            pod_check_data = []
            for t in range(self.model_setup.T):
                pod_check_data.append([
                    t,
                    round(self.S[i, t].x, 2),
                    round(self.U[i, t].x, 2),
                    round(self.model_setup.params['d_pod'][i - self.model_setup.V3[0]][t], 2),
                    round(self.I[i, 2, t].x, 2),
                    round(self.I[i, 3, t].x, 2),
                    round(self.I[i, 4, t].x, 2),
                    round(self.I[i, 5, t].x, 2),
                    round(sum(self.f[j, i, 2, t].x for j in range(self.model_setup.V) if (j, i) in self.model_setup.A), 2),
                    round(sum(self.f[j, i, 3, t].x for j in range(self.model_setup.V) if (j, i) in self.model_setup.A), 2),
                    round(sum(self.f[j, i, 4, t].x for j in range(self.model_setup.V) if (j, i) in self.model_setup.A), 2),
                    round(sum(self.f[j, i, 5, t].x for j in range(self.model_setup.V) if (j, i) in self.model_setup.A), 2),
                    round(sum(sum(self.K[j, i, r, t].x for j in range(self.model_setup.V) if (j, i) in self.model_setup.A) for r in range(self.model_setup.R)), 2),
                    round(sum(sum(self.K[i, j, r, t].x for j in range(self.model_setup.V) if (i, j) in self.model_setup.A) for r in range(self.model_setup.R)), 2),
                    round(self.J[i, t].x, 2)
                ])

            pod_check_df = pd.DataFrame(pod_check_data, columns=[
                'Time', 'Satisfied', 'Shortage', 'Demand', 'FoodInv', 'WaterInv', 'CotInv', 'BlanketInv', 'FoodRec', 'WaterRec', 'CotRec', 'BlanketRec', 'TruckIn', 'TruckOut', 'TruckInv'
            ])

            # Save all results to an Excel file
            with pd.ExcelWriter('optimization_results.xlsx') as writer:
                results_df.to_excel(writer, sheet_name='Summary', index=False)
                shelter_df.to_excel(writer, sheet_name='Shelters', index=False)
                pod_df.to_excel(writer, sheet_name='PoDs', index=False)
                shelter_check_df.to_excel(writer, sheet_name='Shelter Check 22', index=False)
                pod_check_df.to_excel(writer, sheet_name='PoD Check 36', index=False)
        else:
            print("Optimal solution not found")
