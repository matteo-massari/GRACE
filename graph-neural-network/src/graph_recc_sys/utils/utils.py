from datetime import datetime


def get_run_name(model, params):
    model_name = model.__class__.__name__
    timestamp = datetime.now().strftime("%Y-%m-%d@%H:%M")
    run_name = f"{timestamp}_{model_name}_ncl:{params.n_clienti}"  # nome run
    return run_name
