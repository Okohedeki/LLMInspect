def fizzbuzz(n):
    """Return "Fizz" for multiples of 3, "Buzz" for multiples of 5,
    "FizzBuzz" for multiples of both, and the number as a string otherwise."""
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    if n % 15 == 0:
        return "FizzBuzz"
    return str(n)
