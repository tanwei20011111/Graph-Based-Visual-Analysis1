import openai

openai.api_key = "sk-42wM9Var727VeKZkE3D43952C091462fA2D2Af5065745063"
openai.api_base = "https://api.xiaoai.plus/v1"


prompt = ("Instruction: Analyze the given HVAC system energy consumption data to identify the cause of anomalies and "
          "recommend solutions. Focus on the most anomalous metrics and their surrounding time points.\n\n"
          "Context: The HVAC dataset was simulated using the Transient System Simulation tool. This HVAC simulation "
          "dataset is collected from a three-story office building where the HVAC system is applied to the cooling "
          "application of this building. Air handling units are installed on each floor, and variable air volume "
          "terminals control the variable flow. The water tank in this HVAC system uses a flow pump to supply chilled "
          "water to the cooling coils. In total, the dataset simulates a twelve-zone HVAC dataset, which includes the "
          "measured values of temperature sensors, control signals, temperature set points, residential thermal "
          "comfort in each zone, the total estimated power, and the status of the system.\n"
          "Metric Explanation of symbols： T: temperature, sensor_T: sensor temperature reading\n\n"
          "Input Data: The following are the most anomalous metrics and their values at the time of the anomaly, "
          "along with their values two time points before and after the anomaly:\n\n"
          "1. Metric: Power\n"
          "   - Time t-2: 556389.9533\n"
          "   - Time t-1: 572200.1444\n"
          "   - Time t: 229417.0531\n"
          "   - Time t+1: 294624.633\n"
          "   - Time t+2: 353850.7342\n\n"
          "2. Metric: T_chiller\n"
          "   - Time t-2: 9°C\n"
          "   - Time t-1: 9°C\n"
          "   - Time t: 14°C (anomalous)\n"
          "   - Time t+1: 14°C\n"
          "   - Time t+2: 14°C\n\n"
          "3. Metric: sensor_T_chiller\n"
          "   - Time t-2: 9.007°C\n"
          "   - Time t-1: 9.007°C\n"
          "   - Time t: 11.622°C (anomalous)\n"
          "   - Time t+1: 12.507°C\n"
          "   - Time t+2: 13.601°C\n\n"
          "Output Indicator: The answer is what abnormality has occurred in the HVAC system at this moment, the cause, "
          "and the solution.\n"
          "Format:\n"
          "Answer:\n"
          "abnormal:\n"
          "Possible Cause: Briefly explain it in one sentence.\n"
          "Recommendation: Briefly explain it in one sentence.")


completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user",
"content": prompt}])
print(completion.choices[0].message.content)