#ifndef VEC2_HPP
#define VEC2_HPP

#include <iostream>
#include <stdexcept>

// Class template for manipulating 2 dimensional vectors
template <typename T>
class Vec2 {
   public:
    static const Vec2<T> unitX;
    static const Vec2<T> unitY;
    static const Vec2<T> zero;

    T x, y;

    /**
     * @brief Construct a new Vec2 object
     */
    Vec2(T x = 0, T y = 0) : x(x), y(y) {}

    /**
     * @brief The length of the vector
     */
    T length() const { return sqrt(x * x + y * y); }

    /**
     * @brief The squared length of the vector, suitable for comparisons
     * as its more efficient than length()
     */
    T lengthSq() const { return x * x + y * y; }

    /**
     * @brief Vector with the same direction, but length = 1
     */
    Vec2<T> normalized() const {
        T len = length();
        if (len == 0) throw std::runtime_error("Cannot normalize zero vector");
        return Vec2<T>(x / len, y / len);
    }

    // ==================== Operator overloading ====================

    Vec2& operator+=(const Vec2& rhs) {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }

    Vec2& operator-=(const Vec2& rhs) {
        x -= rhs.x;
        y -= rhs.y;
        return *this;
    }

    Vec2& operator*=(const T& rhs) {
        x *= rhs;
        y *= rhs;
        return *this;
    }

    Vec2& operator/=(const T& rhs) {
        x /= rhs;
        y /= rhs;
        return *this;
    }

    bool operator==(const Vec2& rhs) const { return x == rhs.x && y == rhs.y; }

    bool operator!=(const Vec2& rhs) const { return !(*this == rhs); }
};

template <typename T>
const Vec2<T> Vec2<T>::unitX = Vec2<T>(1, 0);

template <typename T>
const Vec2<T> Vec2<T>::unitY = Vec2<T>(0, 1);

template <typename T>
const Vec2<T> Vec2<T>::zero = Vec2<T>(0, 0);

/**
 * @brief Dot product of two 2D vectors
 * @return A new dot product value
 */
template <typename T>
T dot(const Vec2<T>& lhs, const Vec2<T>& rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y;
}

/**
 * @brief Cross product of two 2D vectors
 * @return A new cross product 2D vector
 */
template <typename T>
T cross(const Vec2<T>& lhs, const Vec2<T>& rhs) {
    return lhs.x * rhs.y - lhs.y * rhs.x;
}

template <typename T>
Vec2<T> operator+(const Vec2<T>& lhs, const Vec2<T>& rhs) {
    return Vec2<T>(lhs.x + rhs.x, lhs.y + rhs.y);
}

template <typename T>
Vec2<T> operator-(const Vec2<T>& lhs, const Vec2<T>& rhs) {
    return Vec2<T>(lhs.x - rhs.x, lhs.y - rhs.y);
}

template <typename T>
Vec2<T> operator*(const Vec2<T>& lhs, const T& rhs) {
    return Vec2<T>(lhs.x * rhs, lhs.y * rhs);
}

template <typename T>
Vec2<T> operator/(const Vec2<T>& lhs, const T& rhs) {
    return Vec2<T>(lhs.x / rhs, lhs.y / rhs);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Vec2<T>& vec) {
    return os << "(" << vec.x << ", " << vec.y << ")";
}

#endif  // VEC2_HPP
