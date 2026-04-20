def _test_names(cfg):
    datasets_cfg = getattr(cfg, "DATASETS", None)
    names = getattr(datasets_cfg, "TEST", ()) if datasets_cfg is not None else ()
    if isinstance(names, str):
        return (names,)
    return tuple(names)


def resolve_eval_dataset_name(cfg, dataset_name=None):
    if dataset_name is not None:
        return str(dataset_name)
    names = _test_names(cfg)
    return str(names[0]) if names else None


def _scalar_id(pid):
    if hasattr(pid, "item"):
        try:
            return pid.item()
        except ValueError:
            pass
    return pid


def _all_unavailable(pids):
    if not pids:
        return False
    for pid in pids:
        try:
            if int(_scalar_id(pid)) >= 0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _unique_ids(pids):
    return set(_scalar_id(pid) for pid in pids)


def loader_has_dummy_or_unavailable_ids(val_loader, num_query):
    dataset = getattr(val_loader, "dataset", None)
    items = getattr(dataset, "img_items", None)
    if not items:
        return False

    pids = [item[1] for item in items]
    query_pids = pids[:num_query]
    gallery_pids = pids[num_query:]
    if not query_pids or not gallery_pids:
        return False

    if _all_unavailable(query_pids) or _all_unavailable(gallery_pids):
        return True

    query_ids = _unique_ids(query_pids)
    gallery_ids = _unique_ids(gallery_pids)
    return len(query_ids) <= 1 and len(gallery_ids) <= 1


def should_skip_eval_if_dummy_ids(cfg, val_loader, num_query, dataset_name=None):
    test_cfg = getattr(cfg, "TEST", None)
    if not bool(getattr(test_cfg, "SKIP_EVAL_IF_DUMMY_IDS", True)):
        return False

    resolved_name = resolve_eval_dataset_name(cfg, dataset_name)
    if resolved_name != "UrbanElementsReID_test":
        return False

    return loader_has_dummy_or_unavailable_ids(val_loader, num_query)


def dummy_eval_warning(dataset_name=None):
    name = dataset_name or "the test split"
    return (
        "Skipping metric evaluation for {} because query/gallery IDs are "
        "dummy or unavailable; mAP/CMC would be misleading and will not be "
        "logged. Submission generation is unaffected."
    ).format(name)
