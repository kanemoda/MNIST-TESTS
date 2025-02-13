.PHONY: all menu clean

# List of subproject directories.
SUBPROJECTS = OptiMNIST DeepMNIST ClassicMNIST

# Default target: show the menu.
all: menu

menu:
	@echo "Available projects:"; \
	i=1; \
	for d in $(SUBPROJECTS); do \
	  echo "$$i. $$d"; \
	  i=`expr $$i + 1`; \
	done; \
	echo -n "Enter the number of the project to run: "; \
	read choice; \
	case $$choice in \
	  1) $(MAKE) -C OptiMNIST run ;; \
	  2) $(MAKE) -C DeepMNIST run ;; \
	  3) $(MAKE) -C ClassicMNIST run ;; \
	  *) echo "Invalid option"; exit 1 ;; \
	esac

clean:
	@for d in $(SUBPROJECTS); do \
	  $(MAKE) -C $$d clean; \
	done
