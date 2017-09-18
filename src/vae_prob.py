def modelling_vae(sess, vae, data, batches, epochs):
    cost = []
    for epoch in range(epochs):
        cost.append(
            [sess.run((vae.optimize, vae.error),
                      feed_dict={data: batches[i]})[1]
             for i in range(len(batches))])
    return cost, vae

def test_esn_vae(sess, vae, data, batches):
    return [sess.run(vae.error, feed_dict={data: batches[i]}) for i in range(len(batches))]