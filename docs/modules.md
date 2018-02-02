# Modules

Modules are simply [pure functions](https://en.wikipedia.org/wiki/Pure_function) now. We highly discourage any side effects, including printing to stdout, caching, or any other stateful action that can escape the bounds of the function. 

Signals should accept a [Recording object](recording.md) and return a Recording object. All other arguments are optional. 

## Making your own module

TODO

