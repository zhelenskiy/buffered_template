#include <type_traits>
#include <utility>
#include <optional>
#include <tuple>

template<class... Fs>
struct all_t {
    template<class... Args>
    constexpr static bool value_t = (Fs::template value_t<Args...> && ...);

    template<auto... Args>
    constexpr static bool value_v = (Fs::template value_v<Args...> && ...);
};

template<class... Fs>
struct any_t {
    template<class... Args>
    constexpr static bool value_t = (Fs::template value_t<Args...> || ...);

    template<auto... Args>
    constexpr static bool value_v = (Fs::template value_v<Args...> || ...);
};

template<class F>
struct not_t {
    template<class... Args>
    constexpr static bool value_t = !F::template value_t<Args...>;

    template<auto... Args>
    constexpr static bool value_v = !F::template value_v<Args...>;
};

template<class... Fs>
struct not_all_t {
    template<class... Args>
    constexpr static bool value_t = not_t<all_t<Fs...>>::template value_t<Args...>;

    template<auto... Args>
    constexpr static bool value_v = not_t<all_t<Fs...>>::template value_v<Args...>;
};

template<class... Fs>
struct none_t {
    template<class... Args>
    constexpr static bool value_t = not_t<any_t<Fs...>>::template value_t<Args...>;

    template<auto... Args>
    constexpr static bool value_v = not_t<any_t<Fs...>>::template value_v<Args...>;
};

template<auto item>
struct constantly_v {
    template<class... Args>
    constexpr static decltype(item) value_t = item;

    template<auto... Args>
    constexpr static decltype(item) value_v = item;
};

template<class T>
struct constantly_t {
    template<class... Args>
    using type_t = T;

    template<auto... Args>
    using type_v = T;
};

struct identity_t {
    template<class Arg>
    using type_t = Arg;

    template<auto Arg>
    constexpr static decltype(Arg) value_v = Arg;
};

template<class F, class... FirstArgs>
struct partial_v_t {
    template<class... OtherArgs>
    constexpr static decltype(F::template value_t<FirstArgs..., OtherArgs...>)
            value_t = F::template value_t<FirstArgs..., OtherArgs...>;
};

template<class F, auto... FirstArgs>
struct partial_v_v {
    template<auto... OtherArgs>
    constexpr static decltype(F::template value_v<FirstArgs..., OtherArgs...>)
            value_v = F::template value_v<FirstArgs..., OtherArgs...>;
};

template<class F, class... FirstArgs>
struct partial_t_t {
    template<class... OtherArgs>
    using type_t = typename F::template type_t<FirstArgs..., OtherArgs...>;
};


template<class F, auto... FirstArgs>
struct partial_t_v {
    template<auto... OtherArgs>
    using type_v = typename F::template type_v<FirstArgs..., OtherArgs...>;
};

template<template<class...> class F>
struct from_non_template_v_t {
    template<class... Args>
    constexpr static decltype(F<Args...>::value) value_t = F<Args...>::value;
};

template<template<auto...> class F>
struct from_non_template_v_v {
    template<auto... Args>
    constexpr static decltype(F<Args...>::value) value_v = F<Args...>::value;
};

template<template<class...> class F>
struct from_non_template_t_t {
    template<class... Args>
    using type_t = typename F<Args...>::type;
};

template<template<auto...> class F>
struct from_non_template_t_v {
    template<auto... Args>
    using type_v = typename F<Args...>::type;
};

template<class F, class... Args>
struct to_non_template_v_t {
    constexpr static decltype(F::template value_t<Args...>) value = F::template value_t<Args...>;
};

template<class F, auto... Args>
struct to_non_template_v_v {
    constexpr static decltype(F::template value_v<Args...>) value = F::template value_v<Args...>;
};

template<class F, class... Args>
struct to_non_template_t_t {
    using type = typename F::template type_t<Args...>;
};

template<class F, auto... Args>
struct to_non_template_t_v {
    using type = typename F::template type_v<Args...>;
};

template<class T>
struct is_t : partial_v_t<from_non_template_v_t<std::is_assignable>, T &> {
    template<auto item>
    constexpr static bool value_v = is_t<T>::template value_t<decltype(item)>;
};

template<class T>
struct is_same_with_t : partial_v_t<from_non_template_v_t<std::is_same>, T> {
    template<auto item>
    constexpr static bool value_v = is_same_with_t<T>::template value_t<decltype(item)>;
};

constexpr auto all = [](auto &&...args) {
    return [=](const auto &... subArgs) { return (args(subArgs...) && ...); };
};

constexpr auto any = [](auto &&...args) {
    return [=](const auto &... subArgs) { return (args(subArgs...) || ...); };
};

constexpr auto not_ = [](auto &&f) {
    return [f](auto &&... items) { return !f(std::forward<decltype(items)>(items)...); };
};

constexpr auto not_all = [](auto &&... fs) {
    return not_(all(std::forward<decltype(fs)>(fs)...));
};

constexpr auto none = [](auto &&... fs) {
    return not_(any(std::forward<decltype(fs)>(fs)...));
};

constexpr auto constantly = [](auto &&item) {
    return [item](const auto &&...) { return item; };
};

constexpr auto identity = [](auto &&item) { return item; };

constexpr auto partial = [](auto &&f, auto &&... first) {
    return [f, first...](auto &&... other) { return f(first..., other...); };
};

template<class T>
constexpr auto is = [](auto x) { return is_t<T>::template value_t<decltype(x)>; };

template<class T>
constexpr auto is_same_with = [](auto x) { return is_same_with_t<T>::template value_t<decltype(x)>; };


template<class T, class F, class = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr decltype(auto) operator|(T &&operand, F &&functor) {
    return functor(std::forward<T>(operand));
}

template<class T, class F, class = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr decltype(auto) operator|(const T &operand, F &&functor) {
    return functor(operand);
}

struct void_tag {
};

template<class... Fs>
constexpr auto opt_fun(Fs &&... functors) {
    return [functors...](auto &&arg) {
        using Arg = decltype(arg);
        using R = decltype((*std::forward<Arg>(arg) | ... | functors));
        if constexpr (std::is_void_v<R>) {
            return arg ? ((*std::forward<Arg>(arg) | ... | functors), void_tag()) : std::optional<void_tag>();
        } else {
            return arg ? (*std::forward<Arg>(arg) | ... | functors) : std::optional<R>();
        }
    };
}

template<class T, class S, class = int>
struct has_common_type : std::false_type {
};

template<class T, class S>
struct has_common_type<T, S, decltype(std::declval<std::common_type_t<T, S>>(), 0)> : std::true_type {
};

template<class T, class S>
constexpr bool has_common_type_v = has_common_type<T, S>::value;

template<class S>
constexpr auto otherwise(S &&other) {
    return [other](auto &&opt) {
        if constexpr (has_common_type_v<decltype(opt), S>) {
            return opt ? *opt : other;
        } else if constexpr (std::is_invocable_v<decltype(other)>) {
            if constexpr (std::is_void_v<decltype(other())>) {
                return opt ? *opt : (other(), void_tag());
            } else {
                return opt ? *opt : other();
            }
        } else {
            static_assert(constantly_v<false>::value_t<S, decltype(opt)>, "There is no handler for the types.");
        }
    };
}


template<class... Fs>
constexpr auto tup_fun(Fs &&... functors) {
    return [functors...](auto &&arg) {
        auto apply_functors = [&](auto &&item) { return (std::forward<decltype(item)>(item) | ... | functors); };
        return std::apply([&](auto &&... items) { return std::make_tuple(apply_functors(items)...); },
                          std::forward<decltype(arg)>(arg));
    };
}

template<class... Params, class F>
constexpr auto reduce_params_types(F &&functor) {
    return [functor](Params... params) { return functor(std::forward<Params>(params)...); };
}


struct else_return_t {
    constexpr bool operator()() const { return true; }
} else_return;

template<class F, class V, class = std::enable_if_t<!std::is_same_v<F, else_return_t>>, class... Other>
auto cond(const F &f, V &&v, Other &&... others) {
    return f() ? v : cond(std::forward<Other>(others)...);
}

template<class V>
constexpr auto cond(else_return_t, V &&v) {
    return v;
}

constexpr auto constexpr_cond = [](auto... other) {
    constexpr auto impl = [](auto recur, auto f, auto v, auto... other) {
        if constexpr (std::is_convertible_v<decltype(f), else_return_t>) {
            return v;
        } else if constexpr (f()) {
            return v;
        } else {
            return recur(recur, other...);
        }
    };
    return impl(impl, other...);
};

template<auto... items>
constexpr auto constexpr_cond_v = [](auto... other) {
    constexpr auto impl = [](auto recur, auto f, auto v, auto... other) {
        if constexpr (std::is_convertible_v<decltype(f), else_return_t>) {
            return v;
        } else if constexpr (f(items...)) {
            return v;
        } else {
            return recur(recur, other...);
        }
    };
    return impl(impl, other...);
};

template<class... Args>
constexpr auto constexpr_cond_t = [](auto... other) {
    constexpr auto impl = [](auto recur, auto f, auto v, auto... other) {
        if constexpr (std::is_convertible_v<decltype(f), else_return_t>) {
            return v;
        } else if constexpr (decltype(f)::template value_t<Args...>) {
            return v;
        } else {
            return recur(recur, other...);
        }
    };
    return impl(impl, other...);
};

template<class T, class S, class = std::enable_if_t<!std::is_same_v<T, else_return_t>>, class... Other>
constexpr auto select(const T &item, const T &cur, S &&value, const Other &... others) {
    return item == cur ? value : select(item, others...);
}

template<class T, class S>
constexpr auto select(const T &, else_return_t, S &&value) {
    return value;
}

template<auto F, class T>
struct k2v_t {
    T value;

    explicit k2v_t(T value) : value(std::move(value)) {}
};


template<auto F, class T>
constexpr auto k2v(T &&item) {
    return k2v_t<F, T>(item);
}

template<auto item, decltype(item) cur, class S, class = std::enable_if_t<!std::is_same_v<decltype(item), else_return_t>>, class... Other>
constexpr auto constexpr_select(const k2v_t<cur, S> &pair, const Other &... others) {
    if constexpr (item == cur) {
        return pair.value;
    } else {
        return constexpr_select<item>(others...);
    }
}

template<auto T, class S>
constexpr auto constexpr_select(else_return_t, S &&pair) {
    return pair;
}


constexpr auto copy = [](auto item) { return item; };
constexpr auto move = [](auto &&item) { return std::move(item); }; // NOLINT(bugprone-move-forwarding-reference)

#include <algorithm>

constexpr auto reverse = [](auto &&cont) {
    std::reverse(cont.begin(), cont.end());
    return cont;
};


template<class KeyFinder>
constexpr auto sort(const KeyFinder &keyFinder) {
    return [=](auto &&cont) {
        std::sort(cont.begin(), cont.end(), [&](const auto &a, const auto &b) { return keyFinder(a) < keyFinder(b); });
        return cont;
    };
}

constexpr auto sort() {
    return [](auto &&cont) {
        return cont | sort([](const auto &item) { return item; });
    };
}

template<class... Args>
constexpr auto rsort(Args &&... args) {
    return [=](auto &&cont) {
        return cont | sort(args...) | reverse;
    };
}


#include <iostream>

auto print(const std::string &sep = " ", std::ostream &out = std::cout) {
    return [&out, sep](const auto &container) {
        auto iter = container.cbegin();
        if (iter != container.cend()) {
            out << *iter;
            for (++iter; iter != container.cend(); ++iter) {
                out << sep << *iter;
            }
        }
        return container;
    };
}

auto println(const std::string &sep = " ", std::ostream &out = std::cout, bool flush = true) {
    return [=, &out](const auto &container) {
        container | print(sep, out);
        if (flush) {
            out << std::endl;
        } else {
            out << '\n';
        }
        return container;
    };
}

template<auto a, auto... other>
struct min {
    constexpr static std::common_type_t<decltype(a), decltype(other)...> value =
            a < min<other...>::value ? a : min<other...>::value;
};

template<auto a>
struct min<a> {
    constexpr static decltype(a) value = a;
};

template<auto a, auto... other>
struct max {
    constexpr static std::common_type_t<decltype(a), decltype(other)...> value =
            a > max<other...>::value ? a : max<other...>::value;
};

template<auto a>
struct max<a> {
    constexpr static decltype(a) value = a;
};

template<auto item>
struct struct_for {
    using type = decltype(item);
    constexpr static type value = item;
};


#include "lambdas.h"

//---
template<bool cond, class, class Else>
struct class_if {
    using type = Else;
};

template<class Then, class Else>
struct class_if<true, Then, Else> {
    using type = Then;
};


template<bool cond, class Then, class Else>
using class_if_t = typename class_if<cond, Then, Else>::type;

template<class Seq, bool nothrow = false>
struct lazy_iterator {
    std::optional<Seq> seq_;

    using iterator_category = typename Seq::iterator_tag;
    using value_type = typename Seq::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type *;
    using reference = class_if_t<
            std::is_base_of_v<std::output_iterator_tag, iterator_category>,
            value_type &,
            const value_type &>;

    constexpr lazy_iterator &operator=(const lazy_iterator &other) noexcept(nothrow) {
        if (this != &other)
            if (other.seq_.has_value())
                seq_.emplace(other.seq_.value());
            else
                seq_.reset();
        return *this;
    }

    constexpr bool
    operator==(const lazy_iterator &other) const noexcept(nothrow) {
        return (seq_.has_value() && other.seq_.has_value() && *seq_ == *other.seq_)
               || (seq_->is_empty() && other.seq_->is_empty());
    }

    constexpr bool operator!=(const lazy_iterator &other) const noexcept(nothrow) {
        return !(*this == other);
    }

    constexpr lazy_iterator // NOLINT(cert-dcl21-cpp)
    operator++(int) const noexcept(nothrow) {
        return seq_.has_value() ? lazy_iterator{*seq_->tail()} : *this;
    }

    constexpr lazy_iterator &
    operator++() noexcept(nothrow) {
        return *this = (*this)++;
    }

    [[nodiscard]] constexpr std::optional<value_type> get_node() const noexcept(nothrow) {
        return seq_.has_value() ? seq_->head() : std::nullopt;
    }

    constexpr value_type operator*() const noexcept(nothrow) {
        return *get_node();
    }
};


template<class T, bool nothrow = false>
class infinite_range_t;

auto infinite_range = [](auto a) {
    return infinite_range_t(a);
};

template<class T, bool nothrow>
class infinite_range_t {
    T start_;
public:
    constexpr explicit infinite_range_t(const T &start) noexcept(nothrow) : start_(start) {}

    [[nodiscard]] constexpr bool is_empty() const noexcept { return false; }

    using value_type = T;
    using iterator_tag = std::forward_iterator_tag;
    using const_iterator = lazy_iterator<infinite_range_t, nothrow>;

    [[nodiscard]] const_iterator begin() const noexcept(nothrow) {
        return const_iterator{*this};
    }

    [[nodiscard]] const_iterator end() const noexcept(nothrow) {
        return const_iterator{std::nullopt};
    }

    constexpr infinite_range_t &
    operator=(const infinite_range_t &other) noexcept(nothrow) {
        if (this != &other) {
            start_ = other.start_;
        }
        return *this;
    }

    [[nodiscard]] constexpr std::optional<value_type> head() const noexcept {
        return start_;
    }

    [[nodiscard]] constexpr std::optional<infinite_range_t> tail() const noexcept(nothrow) {
        auto next = start_;
        return infinite_range(++next);
    }

    constexpr bool operator==(const infinite_range_t &other) const noexcept(nothrow) {
        return start_ == other.start_;
    }

    constexpr bool operator!=(const infinite_range_t &other) const noexcept(nothrow) {
        return !(*this == other);
    }
};

template<class Seq, class Size, bool nothrow = false>
class take_t;

auto take = [](auto n) {
    return [n](auto seq) { return take_t(seq, n); };
};

template<class Seq, class Size, bool nothrow>
class take_t {
    Seq seq_;
    Size n_;
public:
    constexpr take_t(Seq seq, Size n) noexcept(nothrow) : seq_(std::move(seq)), n_(std::max(n, Size(0))) {}

    [[nodiscard]] constexpr bool is_empty() const noexcept(nothrow) { return n_ == 0 || seq_.is_empty(); }

    using value_type = typename Seq::value_type;
    using iterator_tag = typename Seq::iterator_tag;
    using const_iterator = lazy_iterator<take_t, nothrow>;

    [[nodiscard]] constexpr const_iterator begin() const noexcept(nothrow) {
        return const_iterator{*this};
    }

    [[nodiscard]] constexpr const_iterator end() const noexcept(nothrow) {
        return const_iterator{std::nullopt}; // end must be reachable
    }

    [[nodiscard]] constexpr std::optional<value_type> head() const noexcept(nothrow) {
        return n_ > Size(0) ? seq_.head() : std::nullopt;
    }

    [[nodiscard]] constexpr std::optional<take_t> tail() const noexcept(nothrow) {
        if (n_ <= Size(0)) return std::nullopt;
        auto new_n = n_;
        return seq_.tail() | opt_fun(take(--new_n));
    }

    constexpr bool operator==(const take_t &other) const noexcept(nothrow) {
        return (is_empty() && other.is_empty()) || (n_ == other.n_ && seq_ == other.seq_);
    }

    constexpr bool operator!=(const take_t &other) const noexcept(nothrow) {
        return !(*this == other);
    }
};

template<class Seq, class Mapper, bool nothrow = false>
class map_t;

auto map = [](auto mapper) {
    return [mapper](auto seq) { return map_t(seq, mapper); };
};

template<class Seq, class Mapper, bool nothrow>
class map_t {
    Seq seq_;
    Mapper mapper_;
    template<class T, class = int>
    struct i_equal : std::false_type {
    };
    template<class T>
    struct i_equal<T, decltype(std::declval<T>() = std::declval<T>(), 0)> : std::true_type {
    };
public:
    constexpr map_t(Seq seq, const Mapper &mapper) : seq_(std::move(seq)), mapper_(mapper) {}

    [[nodiscard]] constexpr bool is_empty() const noexcept(nothrow) { return seq_.is_empty(); }

    using value_type = typename Seq::value_type;
    using iterator_tag = typename Seq::iterator_tag;
    using const_iterator = lazy_iterator<map_t, nothrow>;

    [[nodiscard]] constexpr const_iterator begin() const noexcept(nothrow) {
        return const_iterator{*this};
    }

    [[nodiscard]] constexpr const_iterator end() const noexcept(nothrow) {
        return const_iterator{std::nullopt};
    }

    [[nodiscard]] constexpr std::optional<value_type> head() const noexcept(nothrow) {
        return seq_.head() | opt_fun(mapper_);
    }

    [[nodiscard]] constexpr std::optional<map_t> tail() const noexcept(nothrow) {
        return seq_.tail() | opt_fun(map(mapper_));
    }

    constexpr bool operator==(const map_t &other) const noexcept(nothrow) {
        if (this == &other) return true;
        if constexpr (i_equal<decltype(mapper_)>::value)
            return mapper_ == other.mapper_ && seq_ == other.seq_;
        return false;
    }

    constexpr bool operator!=(const map_t &other) const noexcept(nothrow) {
        return !(*this == other);
    }
};

constexpr auto range = [](auto start, auto count) { return infinite_range(start) | take(count); };
constexpr auto foreach = [](const auto &f) {
    return [f](auto seq) {
        for (const auto &item: seq) item | f;
    };
};

constexpr auto reduce_init = [](auto init, auto operation) {
    return [=](auto seq) {
        auto ini = init;
        seq | foreach([&](const auto &item) { ini = operation(ini, item); });
        return ini;
    };
};

constexpr auto reduce = [](auto operation) {
    return [=](auto seq) {
        return seq.tail() | opt_fun([&](auto rest) { return rest | reduce_init(*seq.head(), operation); });
    };
};

#include <chrono>

constexpr auto trace = [](const auto &... items) { (std::cout << ... << items) << std::endl; };

template<class Action>
void measure(Action action) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    action();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    trace("Time difference = ", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count(), " ms");
    trace("Time difference = ", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count(), " µs");
    trace("Time difference = ", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count(), " ns");
}

#include <vector>
#include <numeric>

int main() {
    trace("Testing composition");
    for (auto t : range(3, 5) | map(fn1(item * 2))) {
        trace(t);
    }
    trace("Testing reduce");
    trace(*(range(1, 4) | reduce(std::plus<>())));
    trace("Performance tests:");
    int res = 0;
    trace("\t#0:\tJust range - lazy");
    measure([&res] {
        range(1, 1000 * 1000 * 1000) | foreach([&res](auto item) { res += item; });
    });
    trace(res);
    trace("\t#0:\tJust range - eager (with storage)");
    res = 0;
    measure([&res] {
        std::vector<int> a(1000 * 1000 * 1000);
        std::iota(a.begin(), a.end(), 1);
        for (auto item : a) res += item;
    });
    trace(res);
    trace("\t#0:\tJust range - eager (without storage)");
    res = 0;
    measure([&res] { for (int i = 1; i <= 1000 * 1000 * 1000; ++i) res += i; });
    trace(res);
    trace("\t#1:\tBig composition - lazy");
    measure([&res] {
        res = range(1, 1000 * 1000 * 1000)
              | take(1000 * 1000)
              | map(fn1(item * 2))
              | reduce(std::plus<>())
              | otherwise(0);
    });
    trace("\t#1:\tBig composition - eager (with storage, range loop)");
    measure([&res] {
        auto local_res = 0;
        std::vector<int> a(1000 * 1000);
        std::iota(a.begin(), a.end(), 1);
        for (auto item : a) local_res += item * 2;
        res = local_res;
    });
    trace(res);
    trace("\t#1:\tBig composition - eager (with storage, index loop)");
    measure([&res] {
        auto local_res = 0;
        std::vector<int> a(1000 * 1000);
        std::iota(a.begin(), a.end(), 1);
        for (auto i = 0; i < a.size(); ++i) local_res += a[i] * 2; // NOLINT(modernize-loop-convert)
        res = local_res;
    });
    trace(res);
    trace("\t#2:\tBig number of elements - lazy");
    measure([&res] {
        res = range(1, 1000 * 1000 * 1000)
              | reduce(std::plus<>())
              | otherwise(0);
    });
    trace(res);
    trace("\t#2:\tBig number of elements - eager (max performance)");
    measure([&res] {
        res = 0;
//        std::vector<int> v(1000*1000);
        for (int i = 1; i <= 1000 * 1000 * 1000; ++i) res += i;
    });
    trace(res);
    for (int n = 10; n <= 1000 * 1000 * 1000; n *= 10) {
        trace("\t#3:\t", n, " elements - lazy");
        measure([&res, n] {
            res = range(1, n) | reduce(std::plus<>()) | otherwise(0);
        });
        trace(res);
        trace("\t#3:\t", n, " elements - eager (max performance)");
        measure([&res, n] {
            res = 0;
//        std::vector<int> v(1000*1000);
            for (int i = 1; i <= n; ++i) res += i;
        });
        trace(res);
        trace();
    }
    return 0;
}
