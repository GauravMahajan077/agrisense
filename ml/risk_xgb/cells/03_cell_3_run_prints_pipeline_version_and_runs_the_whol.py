# CELL 3 — RUN. Prints PIPELINE_VERSION and runs the whole pipeline.
Pipeline(CFG).run()

# After a full run, download ml/models/risk_xgb_artifact.json + risk_xgb_model.json
# and drop them into the local ml/models/ folder. The /ml/risk endpoint then uses
# risk_xgb.predict() automatically (see ml/risk_xgb/README.md).
