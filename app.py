import streamlit as st
import anthropic
import os
from policyengine_us import Simulation

# Set up the Anthropic client
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def get_explanation(variable, value, computation_log):
    prompt = f"""{anthropic.HUMAN_PROMPT} You are an AI assistant explaining US policy calculations. 
    The user has run a simulation for the variable '{variable}' and got a result of {value}.
    Here's the computation log:
    {computation_log}
    
    Please explain this result in simple terms. Your explanation should:
    1. Briefly describe what {variable} is.
    2. Explain the main factors that led to this result.
    3. Mention any key thresholds or rules that affected the calculation.
    4. If relevant, suggest how changes in input might affect this result.
    
    Keep your explanation concise but informative, suitable for a general audience.

    {anthropic.AI_PROMPT}"""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        return f"Failed to get explanation: {str(e)}"


# Streamlit UI
st.title("Policy Interpreter")

# Input fields
age = st.number_input("Your age", min_value=0, max_value=120, value=40)
income = st.number_input("Your employment income", min_value=0, value=20000)
married = st.checkbox("Are you married?")
num_children = st.number_input(
    "Number of children", min_value=0, max_value=10, value=2
)
state = st.selectbox(
    "State", ["CA", "NY", "TX", "FL", "IL"]
)  # Add more states as needed
variable = st.text_input(
    "Variable to analyze (e.g., snap, eitc)", value="snap"
)

if st.button("Calculate and Explain"):
    # Create the situation dictionary
    situation = {
        "people": {
            "you": {
                "age": {"2023": age},
                "employment_income": {"2023": income},
            }
        },
        "families": {"your family": {"members": ["you"]}},
        "marital_units": {"your marital unit": {"members": ["you"]}},
        "tax_units": {"your tax unit": {"members": ["you"]}},
        "households": {
            "your household": {
                "members": ["you"],
                "state_name": {"2023": state},
            }
        },
    }

    # Define unit names explicitly
    unit_names = {
        "families": "your family",
        "marital_units": "your marital unit",
        "tax_units": "your tax unit",
        "households": "your household",
    }

    # Add spouse if married
    if married:
        situation["people"]["spouse"] = {
            "age": {"2023": age},
            "employment_income": {"2023": 0},
        }
        for unit, unit_name in unit_names.items():
            situation[unit][unit_name]["members"].append("spouse")

    # Add children
    for i in range(num_children):
        child_name = f"child_{i+1}"
        situation["people"][child_name] = {
            "age": {"2023": 5},
            "employment_income": {"2023": 0},
        }
        for unit, unit_name in unit_names.items():
            if unit != "marital_units":  # Don't add children to marital units
                situation[unit][unit_name]["members"].append(child_name)

    # Run simulation
    simulation = Simulation(situation=situation)
    simulation.trace = True
    result = simulation.calculate(variable, 2023)

    # Get computation log
    log = simulation.tracer.computation_log
    log_lines = log.lines(aggregate=False, max_depth=None)
    log_str = "\n".join(log_lines)

    # Get explanation from Claude
    explanation = get_explanation(variable, result[0], log_str)

    # Display results
    st.write(f"{variable}: ${result[0]:.2f}")

    st.subheader("Explanation")
    st.write(explanation)

    st.subheader("Computation Log")
    st.text(log_str)

# You might want to add a spinner or progress bar while waiting for the explanation
