aa = list(jnp.arange(100))
bb = jnp.array(101, dtype=jnp.int32)
def test(aa):
    aa.pop(0)
    aa.append(bb)
    return aa
    
print(test(test(aa)))


# aa = jnp.arange(100)
# %timeit jnp.roll(aa, -1, axis=0)