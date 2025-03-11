import unittest
import logging

logging.disable(logging.CRITICAL)

class UnitTest(unittest.TestCase):
    def _is_test_failed(self):
        return any(filter(lambda event: event[1] is not None and event[0] == self and "AssertionError" in event[1], self._outcome.result.failures))

    def _is_error_in_test(self):
        return any(filter(lambda event: event[1] is not None and event[0] == self, self._outcome.result.errors))

    def tearDown(self) -> None:
        print(f"{self.__class__.__name__}.{self._testMethodName}", end=": ")

        if self._is_test_failed():
            print("TEST FAILED")
            return
        if self._is_error_in_test():
            print("ERROR")
            return
        print("TEST PASSED")

