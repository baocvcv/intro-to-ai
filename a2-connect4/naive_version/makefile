.PHONY: compete strategy compete_debug strategy_debug clean test
all: compete strategy

test: compete strategy
	Compete/Compete so/Strategy.so TestCases/80.so result 10
test_debug: compete_debug strategy_debug
	Compete/Compete so/Strategy.so.d TestCases/100.so result 2

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
