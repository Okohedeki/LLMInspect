from fizzbuzz import fizzbuzz


def test_plain_number():
    assert fizzbuzz(7) == "7"


def test_fizz():
    assert fizzbuzz(9) == "Fizz"


def test_buzz():
    assert fizzbuzz(10) == "Buzz"


def test_fizzbuzz():
    assert fizzbuzz(15) == "FizzBuzz"
    assert fizzbuzz(30) == "FizzBuzz"
