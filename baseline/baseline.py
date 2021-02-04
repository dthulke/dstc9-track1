import sys

if __name__ == "__main__":
    if '--generate' in sys.argv:
        from baseline import generate as main
    elif '--multitask' in sys.argv:
        from baseline import multitask as main
    elif '--embedding' in sys.argv:
        from baseline import embedding_selection as main
    else:
        from baseline import main

    main.main()
