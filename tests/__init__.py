import unittest
import logging

logging.disable(logging.CRITICAL)

class UnitTest(unittest.TestCase):
    def _is_test_failed(self, test):
        return any(filter(lambda event: event[1] is not None and event[0] == test and event[1][0] == AssertionError, self._outcome.errors))

    def _is_error_in_test(self, test):
        return any(filter(lambda event: event[1] is not None and event[0] == test, self._outcome.errors))

    def tearDown(self) -> None:
        print(f"{self.__class__.__name__}.{self._testMethodName}", end=": ")

        if self._is_test_failed(self):
            print("TEST FAILED")
            return
        if self._is_error_in_test(self):
            print("ERROR")
            return
        print("TEST PASSED")

