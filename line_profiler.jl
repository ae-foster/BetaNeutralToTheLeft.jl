using DataStructures

global hold = DefaultDict{String, Real}(0.)

function add_timer(expr::Expr)
    if expr.head == :call
        s = String(Symbol(expr))
        return quote
            t0 = time()
            #println($s)
            val = $expr
            t1 = time()
            hold[$s] += t1-t0
            val
        end
    else
        return expr
    end
end

function add_timers(expr::Any)
    if typeof(expr) == Expr
        new_expr = Expr(expr.head, map(e -> add_timers(e), expr.args)...)
        return add_timer(new_expr)
    else
        return expr
    end
end

open("ntl_main.jl") do f
    s = String(readstring(f))
    expr = parse("begin $s end")

    #println(dump(expr))
    #println(dump(add_timers(expr)))
    eval(add_timers(expr))
    println(sort(collect(hold), by=x->x[2]))

end
