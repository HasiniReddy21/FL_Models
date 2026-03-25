import flwr as fl
import logging

# 🔴 Disable Flower logs
logging.getLogger("flwr").setLevel(logging.ERROR)


class CustomStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):

        print(f"\nRound {server_round}")
        print("Received parameters from clients")

        aggregated_parameters = super().aggregate_fit(server_round, results, failures)

        print("Aggregating weights")

        return aggregated_parameters


def main():

    strategy = CustomStrategy(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
    )

    print("Starting Federated Learning...\n")

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=2),  # you can change rounds
        strategy=strategy
    )

    print("\nTraining complete")


if __name__ == "__main__":
    main()

'''import flwr as fl

strategy = fl.server.strategy.FedAvg(

    fraction_fit=1.0,
    min_fit_clients=2,
    min_available_clients=2,

)

fl.server.start_server(

    server_address="localhost:8080",
    strategy=strategy,

) '''