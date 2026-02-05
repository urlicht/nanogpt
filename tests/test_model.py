import torch
from nanogpt.model import AttentionHead, MultiheadAttention, ModelConfig, MLP, AttentionBlock, NanoGPT
from dataclasses import replace

def _make_config(**overrides):
    config = ModelConfig(
        n_vocab=10,
        n_block=8,
        n_layer=1,
        n_head=2,
        d_emb=4,
        dropout_p=0.0,
    )
    config = replace(config, **overrides)

    return config

class TestAttentionHead:
    def test_attention_head_output_shape(self):
        """attention head output shape"""
        config = _make_config(n_block=6, n_head=1)
        head = AttentionHead(config)

        x = torch.randn(2, 5, config.d_emb)
        output = head(x)

        assert output.shape == (2, 5, config.d_emb)

    def test_attention_head_causal(self):
        """check causal and see if it matches the first token value"""
        config = _make_config(n_block=4)
        head = AttentionHead(config)

        weight = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        with torch.no_grad():
            head.query.weight.copy_(weight)
            head.key.weight.copy_(weight)
            head.value.weight.copy_(weight)

        head.eval()

        x1 = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ]
            ]
        )
        out1 = head(x1)

        x2 = x1.clone()
        x2[0, 2, :2] = torch.tensor([100.0, 200.0])
        out2 = head(x2)

        # check that the first two timesteps are identical
        torch.testing.assert_close(out1[:, :2, :], out2[:, :2, :])

        # check that the third timestep if different (sanity check)
        assert not torch.allclose(out1[:, 2, :], out2[:, 2, :])

class TestMultiheadAttention:
    def test_multihead_output(self):
        config = _make_config(n_block=6)
        head = MultiheadAttention(config)

        x = torch.randn(2, 5, config.d_emb)
        output = head(x)

        assert output.shape == (2, 5, config.d_emb)

    def test_multihead_output_shape_various_configs(self):
        """Test output shape with different configurations"""
        configs = [
            _make_config(n_head=1, d_emb=4, n_block=8),
            _make_config(n_head=4, d_emb=16, n_block=10),
            _make_config(n_head=8, d_emb=64, n_block=32),
        ]
        
        for config in configs:
            mha = MultiheadAttention(config)
            x = torch.randn(3, 7, config.d_emb)
            output = mha(x)
            
            assert output.shape == x.shape
            assert output.shape == (3, 7, config.d_emb)

    def test_multihead_causal_masking(self):
        """Test that multihead attention respects causal masking"""
        config = _make_config(n_block=4, n_head=2)
        mha = MultiheadAttention(config)
        mha.eval()

        x1 = torch.randn(1, 3, config.d_emb)
        out1 = mha(x1)

        # Modify a future token
        x2 = x1.clone()
        x2[0, 2, :] = x1[0, 2, :] + torch.randn_like(x2[0, 2, :])
        
        with torch.no_grad():
            out2 = mha(x2)

        # first token should be unaffected by changes to future tokens
        torch.testing.assert_close(out1[:, 0, :], out2[:, 0, :], atol=1e-5, rtol=1e-5)

    def test_multihead_gradient_flow(self):
        """test that gradients flow through multihead attention"""
        config = _make_config(n_head=2)
        mha = MultiheadAttention(config)
        
        x = torch.randn(2, 4, config.d_emb, requires_grad=True)
        output = mha(x)
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_multihead_gradient_flow_to_parameters(self):
        """test that gradients flow to multihead attention parameters"""
        config = _make_config(n_head=2)
        mha = MultiheadAttention(config)
        
        x = torch.randn(2, 4, config.d_emb)
        output = mha(x)
        loss = output.sum()
        loss.backward()
        
        # Check that all parameters have gradients
        for param in mha.parameters():
            assert param.grad is not None
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))

    def test_multihead_eval_mode_no_dropout(self):
        """test that dropout is disabled in eval mode"""
        config = _make_config(n_head=2, dropout_p=0.5)
        mha = MultiheadAttention(config)
        mha.eval()
        
        x = torch.randn(2, 4, config.d_emb)
        
        # Multiple forward passes should give identical results in eval mode
        with torch.no_grad():
            output1 = mha(x)
            output2 = mha(x)
        
        torch.testing.assert_close(output1, output2)

    def test_multihead_train_mode_dropout_stochastic(self):
        """test that dropout introduces stochasticity in train mode"""
        config = _make_config(n_head=2, dropout_p=0.5)
        mha = MultiheadAttention(config)
        mha.train()
        
        x = torch.randn(2, 4, config.d_emb)
        
        # Multiple forward passes should give different results in train mode with dropout
        output1 = mha(x)
        output2 = mha(x)
        
        # With 50% dropout, it's extremely unlikely they're exactly equal
        assert not torch.allclose(output1, output2)

    def test_multihead_sequence_length_independence(self):
        """test that attention handles different sequence lengths"""
        config = _make_config(n_block=10, n_head=2)
        mha = MultiheadAttention(config)
        
        seq_lengths = [1, 3, 5, 10]
        for seq_len in seq_lengths:
            x = torch.randn(2, seq_len, config.d_emb)
            output = mha(x)
            
            assert output.shape == (2, seq_len, config.d_emb)

    def test_multihead_batch_independence(self):
        """test that batch elements are processed independently"""
        config = _make_config(n_head=2)
        mha = MultiheadAttention(config)
        mha.eval()
        
        # create batch with one random element and one deterministic element
        x = torch.randn(2, 4, config.d_emb)
        
        # process as batch
        with torch.no_grad():
            output_batch = mha(x)
        
        # process individually
        with torch.no_grad():
            output_0 = mha(x[0:1])
            output_1 = mha(x[1:2])
        
        # results should match
        torch.testing.assert_close(output_batch[0:1], output_0)
        torch.testing.assert_close(output_batch[1:2], output_1)

    def test_multihead_projection_layer(self):
        """test that projection layer properly transforms concatenated heads"""
        config = _make_config(n_head=2, d_emb=4)
        mha = MultiheadAttention(config)
        
        # projection layer should output d_emb dimensions
        x = torch.randn(1, 3, config.d_emb)
        output = mha(x)
        
        assert output.shape[-1] == config.d_emb
        # output should not be zero
        assert not torch.allclose(output, torch.zeros_like(output))

class TestMLP:
    def test_mlp_shape(self):
        """test MLP output shape"""
        config = _make_config()
        mlp = MLP(config)
        
        x = torch.randn(2, 5, config.d_emb)
        output = mlp(x)
        
        assert output.shape == x.shape

    def test_mlp_layers_and_width(self):
        """test MLP layers shapes"""
        config = _make_config(d_emb=6, mlp_width_multiplier=3)
        mlp = MLP(config)

        assert len(mlp.network) == 5
        assert mlp.network[0].in_features == config.d_emb
        assert mlp.network[0].out_features == config.d_emb * config.mlp_width_multiplier
        assert mlp.network[2].in_features == config.d_emb * config.mlp_width_multiplier
        assert mlp.network[2].out_features == config.d_emb

    def test_mlp_gradient_flow_input(self):
        """test gradient flow through MLP"""
        config = _make_config()
        mlp = MLP(config)

        x = torch.randn(2, 4, config.d_emb, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_mlp_gradient_flow_params(self):
        """test MLP parameters gradient flow"""
        config = _make_config()
        mlp = MLP(config)

        x = torch.randn(2, 4, config.d_emb)
        output = mlp(x)
        loss = output.sum()
        loss.backward()

        for param in mlp.parameters():
            assert param.grad is not None
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))

class TestAttentionBlock:
    def test_attention_block_output_shape(self):
        """test AttentionBlock output shape"""
        config = _make_config(n_block=6, n_head=2, d_emb=4)
        block = AttentionBlock(config)

        x = torch.randn(2, 5, config.d_emb)
        output = block(x)

        assert output.shape == x.shape

    def test_attention_block_residual_identity_when_zeroed(self):
        """test residual path preserves input when sublayers output zeros"""
        config = _make_config(n_block=6, n_head=2, d_emb=4, dropout_p=0.0)
        block = AttentionBlock(config)

        with torch.no_grad():
            for param in block.mh_attn.parameters():
                param.zero_()
            for param in block.ffn.parameters():
                param.zero_()
            block.ln1.weight.fill_(1.0)
            block.ln1.bias.zero_()
            block.ln2.weight.fill_(1.0)
            block.ln2.bias.zero_()

        x = torch.randn(2, 5, config.d_emb)
        output = block(x)

        torch.testing.assert_close(output, x)

    def test_attention_block_gradient_flow(self):
        """test gradient flow through AttentionBlock"""
        config = _make_config(n_block=6, n_head=2, d_emb=4)
        block = AttentionBlock(config)

        x = torch.randn(2, 5, config.d_emb, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

        for param in block.parameters():
            assert param.grad is not None
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))

class TestNanoGPT:
    def test_nanogpt_forward_logits_shape(self):
        """test NanoGPT forward output shapes without targets"""
        config = _make_config(n_vocab=11, n_block=8, n_layer=2, n_head=2, d_emb=4)
        model = NanoGPT(config)

        x = torch.randint(0, config.n_vocab, (2, 5))
        logits, loss = model(x)

        assert logits.shape == (2, 5, config.n_vocab)
        assert loss is None

    def test_nanogpt_forward_with_targets_loss(self):
        """test NanoGPT forward with targets returns loss and flattened logits"""
        config = _make_config(n_vocab=13, n_block=8, n_layer=2, n_head=2, d_emb=4)
        model = NanoGPT(config)

        x = torch.randint(0, config.n_vocab, (2, 5))
        y = torch.randint(0, config.n_vocab, (2, 5))
        logits, loss = model(x, y_target=y)

        assert logits.shape == (2 * 5, config.n_vocab)
        assert loss is not None
        assert loss.ndim == 0

    def test_nanogpt_generate_length_and_vocab(self):
        """test generate extends sequence and stays within vocab"""
        config = _make_config(n_vocab=17, n_block=4, n_layer=1, n_head=1, d_emb=4, dropout_p=0.0)
        model = NanoGPT(config)
        model.eval()

        x = torch.randint(0, config.n_vocab, (2, 6))
        out = model.generate(x, max_n_token=3)

        assert out.shape == (2, 9)
        assert torch.all(out >= 0)
        assert torch.all(out < config.n_vocab)
