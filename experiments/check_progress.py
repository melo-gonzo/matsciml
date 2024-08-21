import json
import matplotlib.pyplot as plt
import numpy as np

num_jobs = 50
the_models = [
    ["mace_pyg", "chgnet_dgl", "m3gnet_dgl"],
    ["mace_pyg"],
    ["m3gnet_dgl"],
    ["chgnet_dgl"],
]
for models in the_models:
    progress = {folder: {i: -1 for i in range(num_jobs)} for folder in models}
    steps = []
    forces = []

    bad_results = []
    bad_steps = []

    for model in models:
        for job_n in range(num_jobs):
            file = f"./calculator_results/matsciml/ai4mat/Elasticity/{model}/job_n-{job_n}/version_0/mp_tests.json"
            try:
                with open(file) as f:
                    data = json.load(f)["_default"]
                completed = len(data) / 7
                for k, v in data.items():
                    if "steps" in v:
                        steps.append(int(v["steps"]["computed"]))
                    if "max-residual-force" in v:
                        forces.append(float(v["max-residual-force"]["computed"]))
                progress[model][job_n] = completed
            except FileNotFoundError:
                bad_results.append(file)

    colors = ["r", "g", "b"]

    which_models = "all" if len(models) > 1 else models[0]

    plt.figure()
    for idx, (model, job) in enumerate(progress.items()):
        completion_percentages = list(job.values())
        plt.hist(
            np.array(completion_percentages) / 215,
            bins=100,
            alpha=0.75,
            color=colors[idx],
            label=model,
        )
        plt.title(f"Job Completion Percentages for {which_models} Models")
        plt.legend()
        plt.xlabel("Percent Complete")
        plt.ylabel("Number of Jobs")
        plt.savefig(f"./plots/progress_histogram_{which_models}.png")

    plt.figure()
    a = plt.hist(
        steps,
        bins=250,
        alpha=0.75,
        color="k",
    )
    plt.title("Steps Taken to Convergence")
    plt.xlabel("Steps Taken")
    plt.ylabel("Number of Tests")
    counts = a[0]
    counts.sort()
    plt.ylim([0, counts[-2]])
    plt.savefig(f"./plots/steps_histogram_{which_models}.png")
    plt.close("all")

    plt.figure()
    S = np.sort(np.array(steps))
    P = np.array(range(len(S))) / float(len(S))
    plt.plot(S, P, color="k")
    plt.title("CDF of Steps Taken")
    plt.xlabel("Steps Taken")
    plt.ylabel("")
    plt.savefig(f"./plots/steps_cdf_{which_models}.png")
    plt.close("all")
    print(10 * "\n")
    print(which_models)
    for step in range(0, 1100, 100):
        try:
            idx = np.where(S <= step)[0][-1]
            cr = round(P[idx], 3)
        except Exception:
            cr = -1
        print(f"Step: {step}\tConvergence Rate:{cr}")
    print(10 * "\n")

    fns = np.array([steps, forces])
    fns = fns[:, fns[1, :] < 0.01]
    # forces = [-1 if f>0.01 else f for f in forces]
    forces = [f for f in forces if f < 0.01]
    plt.figure()
    a = plt.hist(
        forces,
        bins=100,
        alpha=0.75,
        color="k",
    )
    plt.title("Residual Force At Stopping Point")
    plt.xlabel("Residual Force")
    plt.ylabel("Number of Tests")
    plt.yscale("log")
    plt.savefig(f"./plots/force_histogram_{which_models}.png")
    plt.close("all")

    plt.figure()
    S = np.sort(np.array(forces))
    P = np.array(range(len(S))) / float(len(S))
    plt.plot(S, P, color="k")
    plt.title("CDF Residual Force")
    plt.xlabel("Residual Force")
    plt.ylabel("")
    plt.savefig(f"./plots/force_cdf_{which_models}.png")
    plt.close("all")

    plt.figure()
    a = plt.scatter(fns[0, :], fns[1, :], color="k", alpha=0.5)
    plt.title("When Converge?")
    plt.xlabel("Step")
    plt.ylabel("Force")
    plt.savefig(f"./plots/force_step_scatter_{which_models}.png")
    plt.close("all")
