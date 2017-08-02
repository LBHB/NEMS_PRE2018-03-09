def old_enqueue_models_stuff():
        # Rest of code should never run any more, will look through and delete later.

    combos = list(product(cSelected, mSelected))
    failures = []
    for combo in combos[:queuelimit]:
        cell = combo[0]
        model = combo[1]
        try:
            keyword_test_routine(model)
        except Exception as e:
            print("Error when calling nems.fit_single_model for " + mSelected)
            print(e)
            failures += combo
        try:
            stack = nems.fit_single_model(
                            cellid=cell,
                            batch=bSelected,
                            modelname=model,
                            autoplot=False,
                            )
        except Exception as e:
            print("Error when calling nems.fit_single_model for " + mSelected)
            print(e)
            failures += combo
            continue
        plotfile = stack.quick_plot_save()
        r = (
                session.query(NarfResults)
                .filter(NarfResults.cellid == cell)
                .filter(NarfResults.batch == bSelected)
                .filter(NarfResults.modelname == model)
                .all()
                )
        collist = ['%s'%(s) for s in NarfResults.__table__.columns]
        attrs = [s.replace('NarfResults.', '') for s in collist]
        attrs.remove('id')
        attrs.remove('figurefile')
        attrs.remove('lastmod')
        if not r:
            r = NarfResults()
            r.figurefile = plotfile
            r.username = user.username
            if not user.labgroup == 'SPECIAL_NONE_FLAG':
                if not user.labgroup in r.labgroup:
                    r.labgroup += ', %s'%user.labgroup
            # TODO: assign performance variables from stack.meta
            session.add(r)
        else:
            # TODO: assign performance variables from stack.meta
            r[0].figurefile = plotfile
            r[0].username = user.username
            if not user.labgroup == 'SPECIAL_NONE_FLAG':
                if not user.labgroup in r.labgroup:
                    r.labgroup += ', %s'%user.labgroup
            fetch_meta_data(stack, r[0], attrs)

        session.commit()
        
        # Manually release stack for garbage collection - having memory issues?
        stack = None

    session.close()
    
    if queuelimit and (queuelimit >= len(combos)):
        data = (
                "Queue limit present. The first %d "
                "cell/model combinations have been fitted (all)."%queuelimit
                )
    elif queuelimit and (queuelimit < len(combos)):
        data = (
                "Queue limit exceeded. Some cell/model combinations were "
                "not fit (%d out of %d fit)."%(queuelimit, len(combos))
                )
    else:
        data = "All cell/model combinations have been fit (no limit)."
        
    if failures:
        failures = ["%s, %s\n"%(c[0],c[1]) for c in failures]
        failures = " ".join(failures)
        data += "\n Failed combinations: %s"%failures
    
    return jsonify(data=data)