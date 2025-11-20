# ai_agent/agent.py
from rag_setup import create_rag
from commands import train_model, train_optuna, evaluate_model, predict, get_metrics

def main():
    qa = create_rag()
    print("🤖 ML + RAG Agent ready! Type 'exit' to quit.")

    while True:
        user_input = input("\n>> ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # --- ML commands ---
        if "train optuna" in user_input.lower():
            print(train_optuna())
        elif "train model" in user_input.lower():
            print(train_model())
        elif "evaluate" in user_input.lower():
            print(evaluate_model())
        elif "predict" in user_input.lower():
            x = input("Enter input data (comma-separated): ")
            data = [float(i) for i in x.split(",")]
            print(predict(data))
        elif "metrics" in user_input.lower():
            print(get_metrics())
        else:
            # RAG query fallback
            response = qa.invoke({"query": user_input})["result"]
            print(f"\nAnswer:\n{response}")

if __name__ == "__main__":
    main()
