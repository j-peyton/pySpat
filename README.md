# pySpat

**pySpat** is a foundational Python package for spatial point pattern analysis. It is inspired by the design and functionality of the R package [spatstat](https://spatstat.org/) and is written from the ground up in Python with minimal dependencies.

The goal of pySpat is to provide a fully native, extensible spatial statistics library in Python, beginning with the core structures and validations needed for spatial data handling.

## Current Status: Core Framework 

### Implemented

#### Core Classes
- `Window`: Represents axis-aligned rectangular observation windows.
  - Validates bounds
  - Computes area, width, height
  - Checks whether given points lie inside

- `PointPattern`: Represents 2D spatial point patterns.
  - Stores point coordinates and optional marks
  - Validates points within a window
  - Provides summary and coordinate access methods

#### Testing
- Full `pytest` test suite for:
  - `Window`
  - `PointPattern`
- All core tests currently passing

## Project Structure

```
pyspat/
├── core/
│   ├── pointpattern.py
│   └── window.py
├── tests/
│   ├── test_pointpattern.py
│   └── test_window.py
```

## Design Philosophy
- Minimal dependencies (only `numpy` and `pytest`)
- Transparent, scientific structure
- Designed with clarity and extensibility in mind

## Installation & Usage
Not always updating on PyPi, so build directly from repo.
```bash
# Clone the repository
$ git clone https://github.com/j-peyton/pySpat
$ pip install {your_path}/pySpat
```

## Author
Jack Peyton
April 2025

## License
MIT
