.PHONY: compete strategy compete_debug strategy_debug clean test
all: compete strategy

test:
	Compete/Compete so/Strategy.so TestCases/20.so result 1
test_debug:
	Compete/Compete so/Strategy.so.d TestCases/20.so result 1

compete:
	$(MAKE) -C Compete

strategy:
	$(MAKE) -C Strategy so

debug: compete_debug strategy_debug

compete_debug:
	$(MAKE) -C Compete debug

strategy_debug:
	$(MAKE) -C Strategy debug

clean:
	$(MAKE) -C Strategy clean
	$(MAKE) -C Compete clean
