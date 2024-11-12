from tabulate import tabulate


def ae_sum_script():
    # Adjusted data to fit all metrics in a single row
    data = [
        ["Training", "Deploy.", "PROM on deploy.", "Acc.", "Pre.", "Recall", "F1"],
        [0.836, 0.544, 0.807, "86.8%", "86.0%", "96.2%", "90.8%"]
    ]

    # headers = ["", "Perf. to the Oracle", "", "Perf. to the Oracle", "", "Perf. to the Oracle", "", "PROM performance",
    #            "", "PROM performance", "", "PROM performance", "", "PROM performance"]

    # Generate and print the table
    table = tabulate(data, tablefmt="grid", stralign="center", numalign="center")
    print(table)


# ae_sum_script()